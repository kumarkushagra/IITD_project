import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ModelTester:
    def __init__(self, model, history, val_gen, class_indices, min_size, max_size):
        self.model = model
        self.history = history
        self.val_gen = val_gen
        self.class_indices = class_indices
        self.min_size = min_size
        self.max_size = max_size
        
        # internal shared state (set once per visualization run)
        self._img = None
        self._preds = None
        self._true_class = None

    # ----------------------------
    # Training diagnostics
    # ----------------------------

    def plot_history(self):
        hist = self.history.history

        plt.figure(figsize=(14, 4))

        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(hist.get("accuracy", []), label="train")
        plt.plot(hist.get("val_accuracy", []), label="val")
        plt.title("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(hist["loss"], label="train")
        plt.plot(hist["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()

        # Learning rate (if present)
        if "lr" in hist:
            plt.subplot(1, 3, 3)
            plt.plot(hist["lr"])
            plt.title("Learning Rate")

        plt.tight_layout()
        plt.show()

    def plot_confmat(self):
        y_true = []
        y_pred = []

        for _ in range(len(self.val_gen)):
            x, y = next(self.val_gen)
            preds = self.model.predict(x, verbose=0)

            y_true.extend(np.argmax(y, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        labels = [k for k, _ in sorted(self.class_indices.items(), key=lambda x: x[1])]
        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.show()

    # ----------------------------
    # Internal helper
    # ----------------------------

    def _sample_once(self, spatial_core, min_size=None, max_size=None):
        """
        Sample ONE image, resize once, run ONE forward pass.
        All spatial plots reuse this state.
        """

        if (min_size := self.min_size) is None:
            pass

        if (max_size := self.max_size) is None:
            pass

        import random
        import cv2

        idx_to_class = {v: k for k, v in self.class_indices.items()}

        bx, by = self.val_gen[random.randint(0, len(self.val_gen) - 1)]
        i = random.randint(0, bx.shape[0] - 1)

        img = bx[i]
        self._true_class = idx_to_class[np.argmax(by[i])]

        H = random.randint(min_size, max_size)
        W = random.randint(min_size, max_size)
        self._img = cv2.resize(img, (W, H))

        self._preds = spatial_core.predict(
            self._img[None, ...], verbose=0
        )[0]

    # ----------------------------
    # Spatial visualizations
    # ----------------------------

    def show_overlay_grid(self, threshold=0.4, alpha=0.6):
        """3×3 grid: original + per-class overlays."""
        import cv2

        img = self._img
        preds = self._preds
        idx_to_class = {v: k for k, v in self.class_indices.items()}
        H, W = img.shape[:2]

        COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255)
        ]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        axes[0].imshow(img)
        axes[0].set_title(f"Original ({self._true_class})")
        axes[0].axis("off")

        for c in range(preds.shape[-1]):
            heat = cv2.resize(preds[:, :, c], (W, H))
            mask = heat > threshold
            color = np.array(COLORS[c]) / 255.0

            overlay = img.copy()
            for k in range(3):
                overlay[:, :, k] = np.where(
                    mask,
                    (1 - alpha) * overlay[:, :, k] + alpha * heat * color[k],
                    overlay[:, :, k]
                )

            axes[c + 1].imshow(overlay)
            axes[c + 1].set_title(idx_to_class[c])
            axes[c + 1].axis("off")

        for i in range(preds.shape[-1] + 1, 9):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def show_color_maps(self, cmap="jet"):
        """3×3 grid: original + color heatmaps."""
        import cv2

        img = self._img
        preds = self._preds
        idx_to_class = {v: k for k, v in self.class_indices.items()}
        H, W = img.shape[:2]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        axes[0].imshow(img)
        axes[0].set_title(f"Original ({self._true_class})")
        axes[0].axis("off")

        for c in range(preds.shape[-1]):
            heat = cv2.resize(preds[:, :, c], (W, H))
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

            axes[c + 1].imshow(heat, cmap=cmap)
            axes[c + 1].set_title(idx_to_class[c])
            axes[c + 1].axis("off")

        for i in range(preds.shape[-1] + 1, 9):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def show_raw_maps(self):
        """3×3 grid: original + raw heatmaps."""
        import cv2

        img = self._img
        preds = self._preds
        idx_to_class = {v: k for k, v in self.class_indices.items()}
        H, W = img.shape[:2]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        axes[0].imshow(img)
        axes[0].set_title(f"Original ({self._true_class})")
        axes[0].axis("off")

        for c in range(preds.shape[-1]):
            heat = cv2.resize(preds[:, :, c], (W, H))
            axes[c + 1].imshow(heat, cmap="hot")
            axes[c + 1].set_title(idx_to_class[c])
            axes[c + 1].axis("off")

        for i in range(preds.shape[-1] + 1, 9):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Master call
    # ----------------------------

    def show_all(self, spatial_core):
        """
        Use ONE image.
        Run:
        - show_overlay_grid
        - show_color_maps
        - show_raw_maps
        """
        self._sample_once(spatial_core)
        self.show_overlay_grid()
        self.show_color_maps()
        self.show_raw_maps()
