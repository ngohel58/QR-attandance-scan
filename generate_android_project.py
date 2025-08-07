#!/usr/bin/env python3
"""Generate Android Studio project that runs provided Gradio depth app via Chaquopy."""
from dataclasses import dataclass
from pathlib import Path

GRADLE_PLUGIN = "8.1.2"
CHAQUO_PLUGIN = "14.0.2"
GRADLE_VERSION = "8.0"
PYTHON_PACKAGES = [
    "numpy",
    "pillow",
    "opencv-python",
    "torch",
    "torchvision",
    "transformers",
    "scipy",
    "gradio"
]

gradio_py = r'''import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter
import gradio as gr
from transformers import pipeline

try:
    import cv2
    from cv2 import GaussianBlur, bilateralFilter
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChromoStereoizer:
    """
    Advanced depth estimation with multi-scale fusion, gradient-preserving normalization,
    and edge-aware blending for maximum detail preservation.
    """
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        tile_size: int = 518,  # Smaller tiles for more detail
        overlap_ratio: float = 0.5  # Higher overlap for better blending
    ):
        self.depth_pipe = pipeline("depth-estimation", model=model_name)
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.last_original: Optional[Image.Image] = None
        self.last_depth_norm: Optional[np.ndarray] = None

    def _gaussian_filter(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Numpy-based Gaussian filter implementation."""
        if CV2_AVAILABLE:
            kernel_size = max(3, int(6 * sigma + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), sigma)
        else:
            # Fallback using PIL
            if len(image.shape) == 2:
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
                blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
                return np.array(blurred, dtype=np.float32) / 255.0
            else:
                return image  # Return original if can't process

    def _sobel_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Numpy-based Sobel edge detection."""
        if CV2_AVAILABLE:
            return cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
        else:
            # Simple numpy implementation
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            
            # Pad image
            padded = np.pad(image, 1, mode='edge')
            
            # Apply convolution
            grad_x = np.zeros_like(image)
            grad_y = np.zeros_like(image)
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    grad_x[i, j] = np.sum(region * sobel_x)
                    grad_y[i, j] = np.sum(region * sobel_y)
            
            return np.sqrt(grad_x**2 + grad_y**2)

    def _percentile_normalize(self, depth_map: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
        """Robust normalization using percentiles to handle outliers."""
        low, high = np.percentile(depth_map, [p_low, p_high])
        normalized = np.clip((depth_map - low) / max(high - low, 1e-6), 0, 1)
        return normalized

    def _extract_high_freq_details(self, tile_depth: np.ndarray, global_depth: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Extract high-frequency details from tile while preserving global structure."""
        # Create low-frequency version of tile
        tile_low = self._gaussian_filter(tile_depth, sigma=sigma)
        global_low = self._gaussian_filter(global_depth, sigma=sigma) 
        
        # Extract high-frequency details
        tile_details = tile_depth - tile_low
        
        # Add details to global depth
        enhanced = global_depth + tile_details * 0.5  # Adjust strength as needed
        return enhanced

    def _histogram_match_local(self, tile_depth: np.ndarray, global_region: np.ndarray, 
                              preserve_details: bool = True) -> np.ndarray:
        """Advanced histogram matching that preserves local details."""
        if preserve_details:
            # Extract details first
            tile_smooth = self._gaussian_filter(tile_depth, sigma=1.5)
            details = tile_depth - tile_smooth
            
            # Match smooth version to global
            matched_smooth = self._histogram_match(tile_smooth, global_region)
            
            # Add back details
            result = matched_smooth + details * 0.7
        else:
            result = self._histogram_match(tile_depth, global_region)
        
        return np.clip(result, 0, 1)

    def _histogram_match(self, source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Match histogram of source to template."""
        source_flat = source.flatten()
        template_flat = template.flatten()
        
        # Get sorted unique values and their indices
        source_values, source_indices = np.unique(source_flat, return_inverse=True)
        template_values = np.unique(template_flat)
        
        # Interpolate template values to match source quantiles
        source_quantiles = np.linspace(0, 1, len(source_values))
        template_quantiles = np.linspace(0, 1, len(template_values))
        
        interp_values = np.interp(source_quantiles, template_quantiles, template_values)
        
        # Map source values to interpolated template values
        matched_flat = interp_values[source_indices]
        return matched_flat.reshape(source.shape)

    def _edge_aware_blend(self, tile: np.ndarray, global_region: np.ndarray, 
                         weight_map: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
        """Edge-aware blending that preserves sharp transitions."""
        # Modify weights based on edges
        edge_threshold = 0.1
        edge_weights = np.where(edge_map > edge_threshold, 0.8, weight_map)
        
        # Blend with edge awareness
        blended = tile * edge_weights + global_region * (1 - edge_weights)
        return blended

    def _create_seamless_weights(self, h: int, w: int, blend_width: int = 32) -> np.ndarray:
        """Create seamless blending weights with smooth transitions."""
        weights = np.ones((h, w), dtype=np.float32)
        
        # Create fade regions at borders
        for i in range(min(blend_width, min(h, w) // 2)):
            alpha = i / blend_width
            # Top and bottom
            if i < h:
                weights[i, :] *= alpha
                weights[-(i+1), :] *= alpha
            # Left and right  
            if i < w:
                weights[:, i] *= alpha
                weights[:, -(i+1)] *= alpha
        
        # Apply smoothing for even better transitions
        weights = self._gaussian_filter(weights, sigma=blend_width/6)
        return weights

    def _guided_filter_simple(self, depth: np.ndarray, guide: np.ndarray, radius: int = 8) -> np.ndarray:
        """Simplified guided filter using bilateral filtering concept."""
        if CV2_AVAILABLE:
            # Use bilateral filter as approximation
            depth_uint8 = (depth * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(depth_uint8, radius, 50, 50)
            return filtered.astype(np.float32) / 255.0
        else:
            # Fallback to Gaussian filter
            return self._gaussian_filter(depth, sigma=radius/3)

    def generate_depth_map(self, img: Image.Image, mode: str) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Enhanced depth map generation with multiple processing modes."""
        if img is None:
            self.last_original = None
            self.last_depth_norm = None
            return None, None

        self.last_original = img
        W, H = img.size
        
        # Convert to numpy for edge detection
        img_gray = np.array(img.convert('L'), dtype=np.float32) / 255.0

        # 1. Generate global depth map
        try:
            result_global = self.depth_pipe(img)
            raw_global = np.array(result_global["depth"], dtype=np.float32)
            if CV2_AVAILABLE:
                raw_global = cv2.resize(raw_global, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                pil_global = Image.fromarray(raw_global)
                pil_global = pil_global.resize((W, H), resample=Image.BILINEAR)
                raw_global = np.array(pil_global, dtype=np.float32)
        except Exception as e:
            logger.error(f"Global depth inference failed: {e}")
            return None, None

        # Normalize global depth
        global_normalized = self._percentile_normalize(raw_global)

        if mode == "Enhanced Tiled":
            final_depth = self._process_enhanced_tiled(img, img_gray, global_normalized, W, H)
        elif mode == "Multi-Scale Fusion":
            final_depth = self._process_multiscale_fusion(img, img_gray, global_normalized, W, H)
        else:
            final_depth = global_normalized

        self.last_depth_norm = final_depth
        depth_img = Image.fromarray((final_depth * 255).astype(np.uint8))

        # Default effect
        chromo = self.apply_effect(50, 50, 10, 50, 50, 50, 0, 100, 0)
        return depth_img.convert('RGB'), chromo

    def _process_enhanced_tiled(self, img: Image.Image, img_gray: np.ndarray, 
                               global_depth: np.ndarray, W: int, H: int) -> np.ndarray:
        """Enhanced tiled processing with advanced blending."""
        # Edge detection for guidance
        edges = self._sobel_edge_detection(img_gray)
        
        # Initialize accumulators
        accum = np.zeros((H, W), dtype=np.float32)
        weight_total = np.zeros((H, W), dtype=np.float32)
        
        ts = self.tile_size
        stride = int(ts * (1 - self.overlap_ratio))
        
        # Generate tile positions with better coverage
        x_positions = list(range(0, W - ts + 1, stride))
        y_positions = list(range(0, H - ts + 1, stride))
        
        # Ensure edge coverage
        if len(x_positions) == 0 or x_positions[-1] + ts < W:
            x_positions.append(max(0, W - ts))
        if len(y_positions) == 0 or y_positions[-1] + ts < H:
            y_positions.append(max(0, H - ts))

        processed_tiles = 0
        total_tiles = len(x_positions) * len(y_positions)
        
        for y in y_positions:
            for x in x_positions:
                processed_tiles += 1
                logger.info(f"Processing tile {processed_tiles}/{total_tiles} at ({x},{y})")
                
                # Extract tile region
                x_end, y_end = min(x + ts, W), min(y + ts, H)
                tile_w, tile_h = x_end - x, y_end - y
                
                if tile_w <= 0 or tile_h <= 0:
                    continue
                
                # Crop image tile
                tile_img = img.crop((x, y, x_end, y_end))
                
                # Pad if necessary
                if tile_w != ts or tile_h != ts:
                    # Calculate mean color for padding
                    tile_array = np.array(tile_img)
                    mean_color = tuple(map(int, np.mean(tile_array.reshape(-1, tile_array.shape[-1]), axis=0)))
                    
                    padded_tile = Image.new('RGB', (ts, ts), color=mean_color)
                    padded_tile.paste(tile_img, (0, 0))
                    tile_img = padded_tile

                # Process tile
                try:
                    tile_result = self.depth_pipe(tile_img)
                    tile_raw = np.array(tile_result["depth"], dtype=np.float32)
                    
                    # Extract valid region
                    tile_depth = tile_raw[:tile_h, :tile_w]
                    
                    # Get corresponding global region
                    global_region = global_depth[y:y_end, x:x_end]
                    edge_region = edges[y:y_end, x:x_end]
                    
                    # Advanced normalization with detail preservation
                    tile_normalized = self._histogram_match_local(
                        self._percentile_normalize(tile_depth), 
                        global_region, 
                        preserve_details=True
                    )
                    
                    # Multi-scale fusion
                    tile_enhanced = self._extract_high_freq_details(
                        tile_normalized, global_region, sigma=1.5
                    )
                    
                    # Create advanced weight map
                    weight_map = self._create_seamless_weights(
                        tile_h, tile_w, 
                        blend_width=min(32, min(tile_h, tile_w)//4)
                    )
                    
                    # Edge-aware blending
                    tile_final = self._edge_aware_blend(
                        tile_enhanced, global_region, weight_map, edge_region
                    )
                    
                    # Accumulate
                    accum[y:y_end, x:x_end] += tile_final * weight_map
                    weight_total[y:y_end, x:x_end] += weight_map

                except Exception as e:
                    logger.error(f"Tile processing failed at ({x},{y}): {e}")
                    # Use global region as fallback
                    fallback_weight = np.ones((tile_h, tile_w), dtype=np.float32) * 0.1
                    accum[y:y_end, x:x_end] += global_depth[y:y_end, x:x_end] * fallback_weight
                    weight_total[y:y_end, x:x_end] += fallback_weight
                    continue

        # Final blend
        final_depth = np.divide(accum, weight_total, out=global_depth.copy(), where=weight_total > 0)
        
        # Post-processing with guided filtering
        final_depth = self._guided_filter_simple(final_depth, img_gray, radius=4)
        
        return np.clip(final_depth, 0, 1)

    def _process_multiscale_fusion(self, img: Image.Image, img_gray: np.ndarray, 
                                  global_depth: np.ndarray, W: int, H: int) -> np.ndarray:
        """Multi-scale depth fusion for maximum detail."""
        scales = [0.5, 0.75, 1.0, 1.25]  # Different processing scales
        fused_depth = global_depth.copy()
        
        for scale in scales:
            if scale == 1.0:
                continue
                
            # Resize image
            new_w, new_h = int(W * scale), int(H * scale)
            if new_w < 64 or new_h < 64:  # Skip very small scales
                continue
                
            logger.info(f"Processing scale {scale}")
            scaled_img = img.resize((new_w, new_h), Image.BILINEAR)
            
            try:
                # Process at this scale
                scale_result = self.depth_pipe(scaled_img)
                scale_depth = np.array(scale_result["depth"], dtype=np.float32)
                
                # Resize back to original
                if CV2_AVAILABLE:
                    scale_depth = cv2.resize(scale_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    scale_pil = Image.fromarray(scale_depth)
                    scale_depth = np.array(scale_pil.resize((W, H), Image.BILINEAR), dtype=np.float32)
                
                # Normalize and extract details
                scale_normalized = self._percentile_normalize(scale_depth)
                details = scale_normalized - self._gaussian_filter(scale_normalized, sigma=2.0)
                
                # Add scaled details to fusion
                detail_strength = 0.3 / len(scales)  # Adjust strength
                fused_depth += details * detail_strength
                
            except Exception as e:
                logger.error(f"Multi-scale processing failed at {scale}: {e}")
                continue
        
        return np.clip(fused_depth, 0, 1)

    def apply_effect(self, threshold_perc, depth_scale, feather_perc,
                    red_b, blue_b, gamma_perc, black_perc, white_perc, smooth_perc) -> Optional[Image.Image]:
        """Enhanced chromostereopsis effect with better depth mapping."""
        if self.last_original is None or self.last_depth_norm is None:
            return None
            
        gray = np.array(self.last_original.convert('L'), dtype=np.float32)
        
        # Enhanced brightness/contrast adjustment
        black = black_perc * 2.55
        white = white_perc * 2.55
        adj = np.clip((gray - black) / max(white - black, 1e-6), 0, 1)
        
        # Improved gamma correction
        gamma_v = 0.1 + (gamma_perc / 100.0) * 2.9
        adj = np.clip(adj ** gamma_v, 0, 1)
        
        # Enhanced depth processing
        depth_sm = self.last_depth_norm
        if smooth_perc > 0:
            sigma = smooth_perc / 100.0 * 3.0
            depth_sm = self._gaussian_filter(depth_sm, sigma=sigma)
        
        # Better depth mapping with multiple thresholds
        thr = threshold_perc / 100.0
        steep = max(depth_scale, 1e-3) / (feather_perc / 100.0 * 10 + 1)
        
        # Create smoother blend with better falloff
        blend = 1.0 / (1.0 + np.exp(-steep * (depth_sm - thr)))
        
        # Enhanced color mapping
        r = np.clip((red_b / 50.0) * adj * blend * 255, 0, 255).astype(np.uint8)
        b = np.clip((blue_b / 50.0) * adj * (1 - blend) * 255, 0, 255).astype(np.uint8)
        
        # Create output with better color balance
        h, w = r.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[..., 0] = r  # Red channel
        out[..., 2] = b  # Blue channel
        
        return Image.fromarray(out, 'RGB')

    def update_effect(self, *args):
        return self.apply_effect(*args)

    def clear(self):
        self.last_original = None
        self.last_depth_norm = None
        return None, None

# Enhanced UI
stereo = EnhancedChromoStereoizer()

def create_demo():
    with gr.Blocks(title='Enhanced ChromoStereoizer Pro') as demo:
        gr.Markdown('## Enhanced ChromoStereoizer Pro - Maximum Detail Depth Processing')
        gr.Markdown('*Advanced tiled processing with multi-scale fusion and edge-aware blending*')
        
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type='pil', label='Upload Image')
                mode = gr.Radio([
                    'Standard', 
                    'Enhanced Tiled', 
                    'Multi-Scale Fusion'
                ], value='Enhanced Tiled', label='Processing Mode')
                
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("**Processing Parameters**")
                    tile_size_info = gr.Markdown("Tile Size: 384px (optimized for detail)")
                    overlap_info = gr.Markdown("Overlap: 75% (optimized for seamless blending)")
                
                btn = gr.Button('Generate Depth Map', variant='primary')
                
            with gr.Column(scale=1):
                d_out = gr.Image(type='pil', interactive=False, show_download_button=True, label='Depth Map')
                c_out = gr.Image(type='pil', interactive=False, show_download_button=True, label='Chromostereopsis Effect')
                
                with gr.Accordion("Effect Controls", open=True):
                    sliders = [
                        gr.Slider(0, 100, 50, label='Depth Threshold'),
                        gr.Slider(0, 100, 50, label='Depth Scale'),
                        gr.Slider(0, 100, 10, label='Edge Feather'),
                        gr.Slider(0, 100, 50, label='Red Intensity'),
                        gr.Slider(0, 100, 50, label='Blue Intensity'),
                        gr.Slider(0, 100, 50, label='Gamma'),
                        gr.Slider(0, 100, 0, label='Black Level'),
                        gr.Slider(0, 100, 100, label='White Level'),
                        gr.Slider(0, 100, 0, label='Smooth Factor')
                    ]
                
                clr = gr.Button('Clear', variant='secondary')

        # Event handlers
        btn.click(
            lambda m, i: stereo.generate_depth_map(i, m),
            [mode, inp],
            [d_out, c_out],
            show_progress=True
        )
        
        for slider in sliders:
            slider.change(stereo.update_effect, sliders, c_out)
        
        clr.click(stereo.clear, [], [d_out, c_out])
    return demo

demo = create_demo()

def start():
    demo.queue().launch(server_name='0.0.0.0', server_port=7860, share=False, inbrowser=False, show_api=False, debug=False, prevent_thread_lock=True)

if __name__ == '__main__':
    start()
'''

@dataclass
class ProjectGen:
    name: str = "DepthEstimationApp"
    package: str = "com.depth.estimation"

    def __post_init__(self):
        self.pkg_path = self.package.replace('.', '/')
        self.base = Path(self.name)

    def write(self, rel: str, content: str):
        path = self.base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def generate(self):
        self.write('settings.gradle', f"rootProject.name = '{self.name}'\ninclude ':app'\n")
        self.write('build.gradle', f"""buildscript {{
    repositories {{ google(); mavenCentral() }}
    dependencies {{
        classpath 'com.android.tools.build:gradle:{GRADLE_PLUGIN}'
        classpath 'com.chaquo.python:gradle:{CHAQUO_PLUGIN}'
    }}
}}
allprojects {{ repositories {{ google(); mavenCentral() }} }}
task clean(type: Delete) {{ delete rootProject.buildDir }}
""")
        self.write('gradle.properties', 'org.gradle.jvmargs=-Xmx4096m -Dfile.encoding=UTF-8\nandroid.useAndroidX=true\nandroid.enableJetifier=true\n')
        self.write('local.properties', 'sdk.dir=/path/to/Android/Sdk\n')
        self.write('gradle/wrapper/gradle-wrapper.properties', f"""distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https://services.gradle.org/distributions/gradle-{GRADLE_VERSION}-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
""")
        pip_lines = '\n                '.join([f'install "{p}"' for p in PYTHON_PACKAGES])
        self.write('app/build.gradle', f"""plugins {{
    id 'com.android.application'
    id 'com.chaquo.python'
}}

android {{
    namespace '{self.package}'
    compileSdk 34

    defaultConfig {{
        applicationId '{self.package}'
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName '1.0'

        python {{
            buildPython '/usr/bin/python3'
            pip {{
                {pip_lines}
            }}
        }}
    }}

    buildTypes {{
        release {{
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }}
    }}
}}

dependencies {{
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.webkit:webkit:1.7.0'
}}
""")
        manifest = f"""<manifest xmlns:android='http://schemas.android.com/apk/res/android'>
    <uses-permission android:name='android.permission.INTERNET'/>
    <application android:label='@string/app_name' android:icon='@mipmap/ic_launcher' android:usesCleartextTraffic='true'>
        <activity android:name='.MainActivity' android:exported='true'>
            <intent-filter>
                <action android:name='android.intent.action.MAIN'/>
                <category android:name='android.intent.category.LAUNCHER'/>
            </intent-filter>
        </activity>
    </application>
</manifest>
"""
        self.write('app/src/main/AndroidManifest.xml', manifest)
        main_java = f"""package {self.package};

import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.appcompat.app.AppCompatActivity;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {{
    @Override
    protected void onCreate(Bundle savedInstanceState) {{
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        WebView webView = findViewById(R.id.webview);
        webView.getSettings().setJavaScriptEnabled(true);
        webView.setWebViewClient(new WebViewClient());
        if (!Python.isStarted()) {{
            Python.start(new AndroidPlatform(this));
        }}
        Python.getInstance().getModule("gradio_app").callAttr("start");
        webView.loadUrl("http://127.0.0.1:7860/");
    }}
}}
"""
        self.write(f'app/src/main/java/{self.pkg_path}/MainActivity.java', main_java)
        layout = """<?xml version='1.0' encoding='utf-8'?>
<WebView xmlns:android='http://schemas.android.com/apk/res/android'
    android:id='@+id/webview'
    android:layout_width='match_parent'
    android:layout_height='match_parent'/>
"""
        self.write('app/src/main/res/layout/activity_main.xml', layout)
        strings = """<?xml version='1.0' encoding='utf-8'?>
<resources>
    <string name='app_name'>Depth Estimation Pro</string>
</resources>
"""
        self.write('app/src/main/res/values/strings.xml', strings)
        self.write('app/src/main/python/gradio_app.py', gradio_py)
        self.write('app/proguard-rules.pro', "-keep class com.chaquo.python.** { *; }\n")
        print('Project generated at', self.base.resolve())

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='DepthEstimationApp')
    p.add_argument('--package', default='com.depth.estimation')
    args = p.parse_args()
    ProjectGen(args.name, args.package).generate()
