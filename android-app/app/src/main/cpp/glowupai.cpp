#include <jni.h>
#include <android/log.h>

#include <llama.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char *TAG = "glowupai";

llama_model *g_model = nullptr;
llama_context *g_ctx = nullptr;
bool g_backend = false;
std::mutex g_mutex;

void throw_state(JNIEnv *env, const char *message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) {
        env->ThrowNew(cls, message);
        env->DeleteLocalRef(cls);
    }
}

void batch_add(llama_batch &batch, llama_token token, llama_pos pos) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = 1;
    batch.n_tokens++;
}

bool valid_utf8(const std::string &text) {
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(text.c_str());
    while (*bytes != 0) {
        int need;
        if ((*bytes & 0x80) == 0) {
            need = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            need = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            need = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            need = 4;
        } else {
            return false;
        }
        bytes++;
        for (int i = 1; i < need; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes++;
        }
    }
    return true;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_adamxit_glowupai_LlamaEngine_nativeLoad(
        JNIEnv *env, jobject, jstring jpath, jint n_ctx, jint n_threads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model != nullptr && g_ctx != nullptr) {
        return JNI_TRUE;
    }

    if (!g_backend) {
        llama_backend_init();
        g_backend = true;
    }

    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    g_model = llama_load_model_from_file(path, params);
    env->ReleaseStringUTFChars(jpath, path);

    if (g_model == nullptr) {
        throw_state(env, "could not load model");
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    if (g_ctx == nullptr) {
        llama_free_model(g_model);
        g_model = nullptr;
        throw_state(env, "could not create context");
        return JNI_FALSE;
    }

    __android_log_print(ANDROID_LOG_INFO, TAG, "model loaded");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_adamxit_glowupai_LlamaEngine_nativeIsLoaded(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return (g_model != nullptr && g_ctx != nullptr) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_adamxit_glowupai_LlamaEngine_nativeUnload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    if (g_backend) {
        llama_backend_free();
        g_backend = false;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_adamxit_glowupai_LlamaEngine_nativeGenerate(
        JNIEnv *env,
        jobject,
        jstring jprompt,
        jint max_tokens,
        jfloat temp,
        jfloat repeat_penalty,
        jobject callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model == nullptr || g_ctx == nullptr) {
        throw_state(env, "model is not loaded");
        return nullptr;
    }

    jclass cb_class = env->GetObjectClass(callback);
    jmethodID on_token = env->GetMethodID(cb_class, "onToken", "(Ljava/lang/String;)V");
    if (on_token == nullptr) {
        throw_state(env, "token callback is missing");
        return nullptr;
    }

    const char *chars = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt(chars);
    env->ReleaseStringUTFChars(jprompt, chars);

    int32_t needed = llama_tokenize(g_model, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                    nullptr, 0, true, true);
    if (needed < 0) {
        needed = -needed;
    }
    if (needed <= 0) {
        throw_state(env, "could not tokenize prompt");
        return nullptr;
    }

    int32_t n_ctx = llama_n_ctx(g_ctx);
    if (needed + max_tokens > n_ctx) {
        throw_state(env, "prompt exceeds context size");
        return nullptr;
    }

    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    int32_t count = llama_tokenize(g_model, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                   tokens.data(), needed, true, true);
    if (count <= 0) {
        throw_state(env, "could not tokenize prompt");
        return nullptr;
    }
    tokens.resize(static_cast<size_t>(count));

    llama_kv_cache_clear(g_ctx);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    for (int32_t i = 0; i < count; ++i) {
        batch_add(batch, tokens[static_cast<size_t>(i)], i);
        batch.logits[i] = 0;
    }
    batch.logits[count - 1] = 1;

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        throw_state(env, "prompt decode failed");
        return nullptr;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            llama_n_vocab(g_model), llama_token_eos(g_model), llama_token_nl(g_model),
            64, repeat_penalty, 0.0f, 0.0f, false, false));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string output;
    std::string pending;
    std::vector<char> buf(256);

    for (int i = 0; i < max_tokens; ++i) {
        llama_token id = llama_sampler_sample(sampler, g_ctx, -1);
        if (llama_token_is_eog(g_model, id)) {
            break;
        }

        int32_t n = llama_token_to_piece(g_model, id, buf.data(), static_cast<int32_t>(buf.size()), 0, false);
        if (n < 0) {
            buf.resize(static_cast<size_t>(-n));
            n = llama_token_to_piece(g_model, id, buf.data(), static_cast<int32_t>(buf.size()), 0, false);
        }
        if (n > 0) {
            pending.append(buf.data(), static_cast<size_t>(n));
            if (valid_utf8(pending)) {
                output += pending;
                jstring piece = env->NewStringUTF(pending.c_str());
                env->CallVoidMethod(callback, on_token, piece);
                env->DeleteLocalRef(piece);
                pending.clear();
            }
        }

        batch.n_tokens = 0;
        batch_add(batch, id, count + i);
        if (llama_decode(g_ctx, batch) != 0) {
            break;
        }
    }

    if (!pending.empty() && valid_utf8(pending)) {
        output += pending;
        jstring piece = env->NewStringUTF(pending.c_str());
        env->CallVoidMethod(callback, on_token, piece);
        env->DeleteLocalRef(piece);
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return env->NewStringUTF(output.c_str());
}
