import argparse
import cv2
import torch
import numpy as np

from model import SimpleFERNet, load_model
from utils import labels


def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = (face - 0.5) / 0.5
    tensor = torch.from_numpy(face).unsqueeze(0).unsqueeze(0)
    return tensor


def try_downscale_detect(gray, face_cascade, scale=0.5):
    # Run detection on a smaller gray frame and scale boxes back
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    faces = face_cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4)
    boxes = []
    for (x, y, fw, fh) in faces:
        boxes.append((int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best_model.pth')
    parser.add_argument('--camera', type=int, default=0)
  
    parser.set_defaults(flip=True)
    parser.add_argument('--no-flip', dest='flip', action='store_false', help='disable horizontal flip')
    parser.add_argument('--skip', type=int, default=2, help='run inference every N frames (display all)')
    parser.add_argument('--scale', type=float, default=0.5, help='downscale factor for face detection (0.3-1.0)')
    parser.add_argument('--debounce', type=int, default=2, help='consecutive identical predictions required to update displayed emotion')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        net = load_model(args.model, device=device)
    except Exception:
        print('Could not load model from', args.model)
        net = SimpleFERNet()
        net.to(device)
        net.eval()
    use_fp16 = False
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            net.half()
            use_fp16 = True
            print('Using FP16 inference on CUDA')
        except Exception:
            use_fp16 = False

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    try_backends = [None, cv2.CAP_AVFOUNDATION]
    cap = None
    for b in try_backends:
        if b is None:
            cap = cv2.VideoCapture(args.camera)
        else:
            cap = cv2.VideoCapture(args.camera, b)
        if cap is not None and cap.isOpened():
            print('Opened camera', args.camera, 'backend', b)
            break
    if cap is None or not cap.isOpened():
        raise SystemExit('Could not open camera')

    lbls = labels()
    frame_idx = 0
    ema_alpha = 0.6

    # per-face tracking to stabilize labels
    tracks = {}  # id -> {bbox, last_seen, ema_probs, last_pred, last_count, display_label, display_conf}
    next_track_id = 0
    max_track_age = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = try_downscale_detect(gray, face_cascade, scale=max(0.3, min(1.0, args.scale)))

        do_infer = (frame_idx % args.skip == 0)

        # build detections list
        detections = []
        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2
            detections.append({'bbox': (x, y, w, h), 'center': (cx, cy)})

        assigned = {}
        # match detections to existing tracks by nearest center
        for det in detections:
            cx, cy = det['center']
            best_id = None
            best_dist = None
            for tid, t in tracks.items():
                tx, ty, tw, th = t['bbox']
                tcx = tx + tw // 2
                tcy = ty + th // 2
                dist = (tcx - cx) ** 2 + (tcy - cy) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None and best_dist is not None and best_dist < (100 ** 2):
                assigned[best_id] = det['bbox']
            else:
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {'bbox': det['bbox'], 'last_seen': frame_idx, 'ema_probs': None, 'last_pred': None, 'last_count': 0, 'display_label': None, 'display_conf': 0.0}
                assigned[tid] = det['bbox']

        # update tracks
        for tid, bbox in list(assigned.items()):
            tracks[tid]['bbox'] = bbox
            tracks[tid]['last_seen'] = frame_idx

        # remove stale tracks
        for tid in list(tracks.keys()):
            if frame_idx - tracks[tid]['last_seen'] > max_track_age:
                del tracks[tid]

        # inference / draw per track
        for tid, t in list(tracks.items()):
            x, y, w, h = t['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if not do_infer:
                if t['display_label'] is not None:
                    cv2.putText(frame, f"{t['display_label']} {t['display_conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            tensor = preprocess_face(face).to(device)
            if use_fp16:
                tensor = tensor.half()
            with torch.no_grad():
                out = net(tensor)
                probs = torch.nn.functional.softmax(out, dim=1).cpu().squeeze(0)
                if t['ema_probs'] is None:
                    t['ema_probs'] = probs.clone()
                else:
                    t['ema_probs'] = ema_alpha * t['ema_probs'] + (1.0 - ema_alpha) * probs
                conf_val, idx_val = t['ema_probs'].max(0)
                pred_idx = int(idx_val.item())
                pred_label = lbls[pred_idx]

                if t['last_pred'] is None or pred_idx != t['last_pred']:
                    t['last_pred'] = pred_idx
                    t['last_count'] = 1
                else:
                    t['last_count'] += 1

                if t['display_label'] is None:
                    t['display_label'] = pred_label
                    t['display_conf'] = float(conf_val.item())
                else:
                    if pred_idx != lbls.index(t['display_label']):
                        if t['last_count'] >= args.debounce:
                            t['display_label'] = pred_label
                            t['display_conf'] = float(conf_val.item())
                    else:
                        t['display_conf'] = float(conf_val.item())

            if t['display_label'] is not None:
                cv2.putText(frame, f"{t['display_label']} {t['display_conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame_idx += 1

        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
