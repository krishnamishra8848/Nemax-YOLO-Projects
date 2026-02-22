# Employee YOLO Classifier (Streamlit)

This app runs a trained Ultralytics classification model with:
- Realtime webcam prediction (browser camera)
- Voice output for predicted class
- Image upload prediction

## 1) Setup

```bash
python -m pip install -r requirements.txt
```

## 2) Model path

Expected model location:

```text
runs_cls/employee_cls_10ep/weights/best.pt
```

If your model is in a different folder, update `MODEL_PATH` in `app.py`.

## 3) Run app

```bash
python -m streamlit run app.py
```

## Notes

- Webcam uses browser permission (click START and allow camera).
- Voice output is generated on the machine running Streamlit.
- For GitHub push, training artifacts are ignored by `.gitignore` by default.
