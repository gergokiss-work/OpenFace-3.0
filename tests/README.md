# OpenFace-3.0 Live Camera Tests

Test scripts for evaluating OpenFace-3.0 smile detection capabilities with live camera feed.

## Overview

These scripts test the Action Unit (AU) detection from OpenFace-3.0, specifically focusing on:
- **AU6** (Cheek Raiser) - Indicates genuine smile with eye involvement (Duchenne marker)
- **AU12** (Lip Corner Puller) - Primary smile indicator (mouth movement)

## Scripts

### 1. `live_camera_test.py` (V1)
Basic test sending full camera frame to model.
- **Issue discovered**: Model expects cropped face, not full frame

### 2. `live_camera_test_v2.py` (V2)
Added face detection (OpenCV Haar Cascade) to crop face before model inference.
- Shows AU bars and smile detection status
- Includes latency measurements

### 3. `live_camera_test_v3.py` (V3 - Enhanced)
Full-featured display with:
- Camera feed with face bounding box
- Cropped face preview (model input)
- All 8 AU values with visual bars
- Smile detection status box
- All 8 emotion probabilities
- Latency breakdown
- Peak value tracking
- Real-time AU history graph

### 4. `debug_au_values.py`
Debug script to analyze raw AU output values:
- Shows raw values (can be negative)
- Shows values after sigmoid transformation (0-1 range)
- Helps diagnose AU detection issues

## Usage

```bash
cd /Users/gergokiss/work/gergo/OpenFace-3.0
source venv/bin/activate
python tests/live_camera_test_v3.py
```

### Controls
- `Q` - Quit
- `S` - Save current frame with AU values
- `R` - Reset peak values and history
- `SPACE` (debug script) - Capture and analyze single frame

## AU Index Mapping

| Index | AU Code | Description | Smile Relevance |
|-------|---------|-------------|-----------------|
| 0 | AU1 | Inner Brow Raiser | - |
| 1 | AU2 | Outer Brow Raiser | - |
| 2 | AU4 | Brow Lowerer | - |
| 3 | AU6 | Cheek Raiser | **Duchenne smile marker** |
| 4 | AU9 | Nose Wrinkler | - |
| 5 | AU12 | Lip Corner Puller | **Primary smile** |
| 6 | AU25 | Lips Part | Secondary |
| 7 | AU26 | Jaw Drop | Secondary |

## Smile Detection Logic

```
if AU12 < 0.15:
    → "NO SMILE"
elif AU6 < 0.08:
    → "SOCIAL SMILE" (mouth only, no eye involvement)
elif AU6/AU12 >= 0.15:
    → "DUCHENNE SMILE" (genuine smile with eye crinkles)
else:
    → "PARTIAL SMILE"
```

## Technical Notes

### Model Input Requirements
- The MLT model expects a **cropped face image**, not full frame
- Input is resized to 224x224 and normalized with ImageNet stats
- Face detection is required before inference

### AU Output Format
- Raw output can be **negative or positive** (not bounded 0-1)
- The AU_model uses L2 normalization and dot product with class centers
- Sigmoid may need to be applied for 0-1 range interpretation

### Latency Breakdown (typical on CPU)
- Capture: ~15-85ms
- Face Detection: ~10-30ms
- Preprocess: ~5-10ms
- Inference: ~100-120ms
- **Total: ~130-250ms**

## Test Results

### Initial Findings
- AU25 (Lips Part) and AU26 (Jaw Drop) show values
- AU6 and AU12 often show 0 - may need sigmoid activation
- Further investigation needed on raw output interpretation

## Dependencies

```
torch>=2.0.0
torchvision
timm>=1.0.15
opencv-python
numpy<2
Pillow
```

## Related Documentation

- [OPENFACE3_SMILE_INTEGRATION_ASSESSMENT.md](../../sandbox/liveness-poc/qa-testing/docs/OPENFACE3_SMILE_INTEGRATION_ASSESSMENT.md)
- [OPENFACE3_DETAILED_INTEGRATION_PLAN.md](../../sandbox/liveness-poc/qa-testing/docs/OPENFACE3_DETAILED_INTEGRATION_PLAN.md)
