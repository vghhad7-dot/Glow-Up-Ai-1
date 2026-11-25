package com.adamxit.glowupai

object LlamaEngine {

    fun interface TokenCallback {
        fun onToken(token: String)
    }

    init {
        System.loadLibrary("glowupai")
    }

    fun load(modelPath: String, nCtx: Int = 1024, nThreads: Int = 4): Boolean =
        nativeLoad(modelPath, nCtx, nThreads)

    fun generate(
        prompt: String,
        maxTokens: Int = 120,
        temp: Float = 0.7f,
        repeatPenalty: Float = 1.05f,
        callback: TokenCallback
    ): String = nativeGenerate(prompt, maxTokens, temp, repeatPenalty, callback)

    fun unload() = nativeUnload()

    fun isLoaded(): Boolean = nativeIsLoaded()

    private external fun nativeLoad(modelPath: String, nCtx: Int, nThreads: Int): Boolean

    private external fun nativeGenerate(
        prompt: String,
        maxTokens: Int,
        temp: Float,
        repeatPenalty: Float,
        callback: TokenCallback
    ): String

    private external fun nativeUnload()

    private external fun nativeIsLoaded(): Boolean
}
