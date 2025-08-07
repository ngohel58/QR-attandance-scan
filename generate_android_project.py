#!/usr/bin/env python3
"""Simple Android Studio project generator for depth estimation app."""
import os
from pathlib import Path
from dataclasses import dataclass

depth_java_template = r"""package __PACKAGE__;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.*;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_IMAGE_PICK = 2;
    private static final int PERMISSION_REQUEST = 100;

    private ImageView originalImage, depthImage, effectImage;
    private Button selectBtn, captureBtn, processBtn, saveBtn;
    private SeekBar redSeek, blueSeek, threshSeek;
    private TextView redVal, blueVal, threshVal;
    private Python py;
    private PyObject module;
    private Bitmap currentBmp, depthBmp, effectBmp;
    private ProgressDialog progress;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initPython();
        initViews();
        setListeners();
        checkPermissions();
    }

    private void initPython() {
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        py = Python.getInstance();
        module = py.getModule("depth_processor");
    }

    private void initViews() {
        originalImage = findViewById(R.id.originalImage);
        depthImage = findViewById(R.id.depthImage);
        effectImage = findViewById(R.id.effectImage);
        selectBtn = findViewById(R.id.selectImageBtn);
        captureBtn = findViewById(R.id.captureImageBtn);
        processBtn = findViewById(R.id.processBtn);
        saveBtn = findViewById(R.id.saveBtn);
        redSeek = findViewById(R.id.redSeekBar);
        blueSeek = findViewById(R.id.blueSeekBar);
        threshSeek = findViewById(R.id.thresholdSeekBar);
        redVal = findViewById(R.id.redValue);
        blueVal = findViewById(R.id.blueValue);
        threshVal = findViewById(R.id.thresholdValue);
        progress = new ProgressDialog(this);
        progress.setMessage("Processing...");
        progress.setCancelable(false);
    }

    private void setListeners() {
        selectBtn.setOnClickListener(v -> selectImage());
        captureBtn.setOnClickListener(v -> captureImage());
        processBtn.setOnClickListener(v -> processImage());
        saveBtn.setOnClickListener(v -> saveResults());

        SeekBar.OnSeekBarChangeListener listener = new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (seekBar == redSeek) redVal.setText(String.valueOf(progress));
                if (seekBar == blueSeek) blueVal.setText(String.valueOf(progress));
                if (seekBar == threshSeek) threshVal.setText(String.valueOf(progress));
                if (depthBmp != null && fromUser) applyEffect();
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        };
        redSeek.setOnSeekBarChangeListener(listener);
        blueSeek.setOnSeekBarChangeListener(listener);
        threshSeek.setOnSeekBarChangeListener(listener);
    }

    private void checkPermissions() {
        String[] perms = {Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE};
        boolean need = false;
        for (String p : perms) {
            if (ContextCompat.checkSelfPermission(this, p) != PackageManager.PERMISSION_GRANTED) {
                need = true; break;
            }
        }
        if (need) ActivityCompat.requestPermissions(this, perms, PERMISSION_REQUEST);
    }

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_IMAGE_PICK);
    }

    private void captureImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
        }
    }

    private void processImage() {
        if (currentBmp == null) {
            Toast.makeText(this, "Select an image first", Toast.LENGTH_SHORT).show();
            return;
        }
        progress.show();
        new Thread(() -> {
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            currentBmp.compress(Bitmap.CompressFormat.PNG, 100, os);
            byte[] data = os.toByteArray();
            PyObject result = module.callAttr("process_image", data);
            byte[] depthBytes = result.callAttr("get_depth_bytes").toJava(byte[].class);
            depthBmp = BitmapFactory.decodeByteArray(depthBytes, 0, depthBytes.length);
            runOnUiThread(() -> {
                depthImage.setImageBitmap(depthBmp);
                applyEffect();
                progress.dismiss();
            });
        }).start();
    }

    private void applyEffect() {
        float r = redSeek.getProgress()/100f;
        float b = blueSeek.getProgress()/100f;
        float t = threshSeek.getProgress()/100f;
        new Thread(() -> {
            PyObject arr = module.callAttr("apply_effect", r, b, t);
            byte[] effBytes = arr.toJava(byte[].class);
            effectBmp = BitmapFactory.decodeByteArray(effBytes, 0, effBytes.length);
            runOnUiThread(() -> effectImage.setImageBitmap(effectBmp));
        }).start();
    }

    private void saveResults() {
        if (depthBmp == null || effectBmp == null) {
            Toast.makeText(this, "Nothing to save", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            String ts = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "DepthEstimation");
            if (!dir.exists()) dir.mkdirs();
            File depthFile = new File(dir, "depth_"+ts+".png");
            FileOutputStream dOut = new FileOutputStream(depthFile);
            depthBmp.compress(Bitmap.CompressFormat.PNG, 100, dOut);
            dOut.close();
            File effectFile = new File(dir, "effect_"+ts+".png");
            FileOutputStream eOut = new FileOutputStream(effectFile);
            effectBmp.compress(Bitmap.CompressFormat.PNG, 100, eOut);
            eOut.close();
            Toast.makeText(this, "Saved to "+dir.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch (IOException e) {
            Log.e("DepthEstimation", "save", e);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_PICK && data != null) {
                try {
                    Uri uri = data.getData();
                    InputStream stream = getContentResolver().openInputStream(uri);
                    currentBmp = BitmapFactory.decodeStream(stream);
                    originalImage.setImageBitmap(currentBmp);
                } catch (IOException e) { e.printStackTrace(); }
            } else if (requestCode == REQUEST_IMAGE_CAPTURE && data != null) {
                Bundle extras = data.getExtras();
                currentBmp = (Bitmap) extras.get("data");
                originalImage.setImageBitmap(currentBmp);
            }
        }
    }
}
"""


@dataclass
class ProjectGen:
    name: str = "DepthEstimationApp"
    package: str = "com.depth.estimation"

    def __post_init__(self):
        self.pkg_path = self.package.replace('.', '/')
        self.base = Path(self.name)

    def write(self, rel, content):
        path = self.base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def generate(self):
        self.write('build.gradle', """buildscript {
    repositories { google(); mavenCentral() }
    dependencies { classpath 'com.android.tools.build:gradle:8.1.2'; classpath 'com.chaquo.python:gradle:14.0.2' }
}
allprojects { repositories { google(); mavenCentral() } }
""")
        self.write('settings.gradle', f"rootProject.name='{self.name}'\ninclude(':app')\n")
        self.write('app/build.gradle', f"""plugins {{ id 'com.android.application'; id 'com.chaquo.python' }}
android {{ compileSdk 34
    defaultConfig {{ applicationId '{self.package}'; minSdk 24; targetSdk 34; versionCode 1; versionName '1.0'
        python {{ buildPython '/usr/bin/python3'; pip {{ install 'numpy'; install 'pillow'; install 'opencv-python'; install 'torch'; install 'torchvision'; install 'transformers'; install 'scipy' }} }} }}
}}
dependencies {{ implementation 'androidx.appcompat:appcompat:1.6.1' }}
""")
        self.write('app/src/main/AndroidManifest.xml', f"""<manifest xmlns:android='http://schemas.android.com/apk/res/android'>
<application android:label='@string/app_name' android:icon='@mipmap/ic_launcher'>
    <activity android:name='.MainActivity'>
        <intent-filter><action android:name='android.intent.action.MAIN'/><category android:name='android.intent.category.LAUNCHER'/></intent-filter>
    </activity>
</application>
</manifest>
""")
        java_code = depth_java_template.replace('__PACKAGE__', self.package)
        self.write(f'app/src/main/java/{self.pkg_path}/MainActivity.java', java_code)
        self.write('app/src/main/python/depth_processor.py', """import io, numpy as np
from PIL import Image
from transformers import pipeline
model = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas', device='cpu')

def process_image(data):
    img = Image.open(io.BytesIO(data))
    result = model(img)
    depth = result['depth']
    depth_bytes = io.BytesIO()
    depth.save(depth_bytes, format='PNG')
    return depth_bytes.getvalue()

def get_depth_bytes():
    return b''

def apply_effect(r,b,t):
    return b''
""")
        self.write('app/src/main/res/layout/activity_main.xml', '<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android" android:layout_width="match_parent" android:layout_height="match_parent" android:orientation="vertical"/>')
        self.write('app/src/main/res/values/strings.xml', """<resources><string name='app_name'>Depth Estimation Pro</string></resources>""")
        self.write('local.properties', 'sdk.dir=/path/to/Android/Sdk\n')
        print('Project generated at', self.base.resolve())

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='DepthEstimationApp')
    p.add_argument('--package', default='com.depth.estimation')
    args = p.parse_args()
    gen = ProjectGen(args.name, args.package)
    gen.generate()
