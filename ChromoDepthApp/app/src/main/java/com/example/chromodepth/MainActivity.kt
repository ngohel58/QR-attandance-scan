package com.example.chromodepth

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.github.chrisbanes.photoview.PhotoView
import com.microsoft.onnxruntime.*
import java.io.ByteArrayOutputStream
import java.io.InputStream
import kotlin.math.exp

class MainActivity : AppCompatActivity() {
    private lateinit var originalView: PhotoView
    private lateinit var depthView: PhotoView
    private lateinit var effectView: PhotoView
    private lateinit var generateBtn: Button
    private lateinit var applyBtn: Button
    private lateinit var saveBtn: Button
    private lateinit var thresholdSeek: SeekBar

    private var originalBmp: Bitmap? = null
    private var depthBmp: Bitmap? = null
    private var effectBmp: Bitmap? = null

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        originalView = findViewById(R.id.originalView)
        depthView = findViewById(R.id.depthView)
        effectView = findViewById(R.id.effectView)
        generateBtn = findViewById(R.id.generateBtn)
        applyBtn = findViewById(R.id.applyBtn)
        saveBtn = findViewById(R.id.saveBtn)
        thresholdSeek = findViewById(R.id.thresholdSeek)

        ortEnv = OrtEnvironment.getEnvironment()
        val modelBytes = assets.open("model.onnx").readBytes()
        val opts = OrtSession.SessionOptions()
        ortSession = ortEnv!!.createSession(modelBytes, opts)

        val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { loadImage(it) }
        }

        findViewById<Button>(R.id.pickBtn).setOnClickListener { pickImage.launch("image/*") }

        generateBtn.setOnClickListener {
            originalBmp?.let {
                depthBmp = runDepthModel(it)
                depthView.setImageBitmap(depthBmp)
            }
        }

        applyBtn.setOnClickListener {
            val t = thresholdSeek.progress
            if (originalBmp != null && depthBmp != null) {
                effectBmp = applyEffect(originalBmp!!, depthBmp!!, t)
                effectView.setImageBitmap(effectBmp)
            }
        }

        saveBtn.setOnClickListener {
            // TODO: save depthBmp and effectBmp to storage
        }

        thresholdSeek.setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener{
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser && originalBmp != null && depthBmp != null) {
                    effectBmp = applyEffect(originalBmp!!, depthBmp!!, progress)
                    effectView.setImageBitmap(effectBmp)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun loadImage(uri: Uri) {
        val stream: InputStream? = contentResolver.openInputStream(uri)
        originalBmp = BitmapFactory.decodeStream(stream)
        originalView.setImageBitmap(originalBmp)
    }

    private fun runDepthModel(bitmap: Bitmap): Bitmap {
        val w = 1036
        val h = 1036
        val resized = Bitmap.createScaledBitmap(bitmap, w, h, true)
        val inputBuffer = FloatArray(1 * 3 * h * w)
        var idx = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val pixel = resized.getPixel(x, y)
                inputBuffer[idx++] = ((pixel shr 16 and 0xFF) / 255f)
                inputBuffer[idx++] = ((pixel shr 8 and 0xFF) / 255f)
                inputBuffer[idx++] = ((pixel and 0xFF) / 255f)
            }
        }
        val shape = longArrayOf(1, 3, h.toLong(), w.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)
        val output = ortSession!!.run(mapOf("image" to tensor))
        val outArray = (output[0].value as Array<Array<Array<FloatArray>>>)[0][0]
        val depthBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        var maxVal = Float.NEGATIVE_INFINITY
        var minVal = Float.POSITIVE_INFINITY
        for (row in outArray) {
            for (v in row) {
                if (v > maxVal) maxVal = v
                if (v < minVal) minVal = v
            }
        }
        val scale = maxVal - minVal
        for (y in 0 until h) {
            for (x in 0 until w) {
                val v = ((outArray[y][x] - minVal) / scale * 255f).toInt()
                val c = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                depthBmp.setPixel(x, y, c)
            }
        }
        return Bitmap.createScaledBitmap(depthBmp, bitmap.width, bitmap.height, true)
    }

    private fun applyEffect(src: Bitmap, depth: Bitmap, threshold: Int): Bitmap {
        val w = src.width
        val h = src.height
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val thresh = threshold / 100f
        for (y in 0 until h) {
            for (x in 0 until w) {
                val dp = (depth.getPixel(x, y) and 0xFF) / 255f
                val srcPx = src.getPixel(x, y)
                val gray = (0.299f * (srcPx shr 16 and 0xFF) + 0.587f * (srcPx shr 8 and 0xFF) + 0.114f * (srcPx and 0xFF)) / 255f
                val blend = 1f / (1f + exp(-10f * (dp - thresh)))
                val r = (gray * blend * 255).toInt().coerceIn(0,255)
                val b = (gray * (1f - blend) * 255).toInt().coerceIn(0,255)
                result.setPixel(x, y, (0xFF shl 24) or (r shl 16) or b)
            }
        }
        return result
    }
}
