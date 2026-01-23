import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_recall_fscore_support

def evaluate_and_generate_pdf(model,history,test_it,steps_test,model_name,dataset_name,training_time_seconds,pdf_name,img_size,batch_size,optimizer_name,learning_rate,class_weights=None):
    # =========================
    # Evaluation
    # =========================
    test_it.reset()
    test_loss, test_acc = model.evaluate(test_it, steps=steps_test, verbose=0)

    # =========================
    # Predictions (MULTICLASS FIX)
    # =========================
    test_it.reset()
    class_labels = list(test_it.class_indices.keys())

    y_true = test_it.classes
    y_pred_prob = model.predict(test_it, steps=steps_test, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # =========================
    # Metrics
    # =========================
    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    class_report = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        output_dict=True
    )

    # =========================
    # Confidence analysis
    # =========================
    confidence = np.max(y_pred_prob, axis=1)

    # =========================
    # Model info
    # =========================
    model_path = f"{model_name}.keras"
    model_size_mb = (
        os.path.getsize(model_path) / (1024 * 1024)
        if os.path.exists(model_path) else 0
    )
    total_params = model.count_params()

    # =========================
    # PDF REPORT
    # =========================
    with PdfPages(pdf_name) as pdf:

        # ---------- PAGE 1: SUMMARY ----------
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")

        plt.text(
            0.5, 0.95, "MODEL EVALUATION REPORT",
            ha="center", va="center",
            fontsize=18, fontweight="bold"
        )

        summary_table = [
            ["Dataset", dataset_name],
            ["Model", model_name],
            ["Input Size", f"{img_size[0]} × {img_size[1]} × 3"],
            ["Batch Size", batch_size],
            ["Optimizer", optimizer_name],
            ["Learning Rate", learning_rate],
            ["Training Time (s)", f"{training_time_seconds:.2f}"],
            ["Model Size (MB)", f"{model_size_mb:.2f}"],
            ["Total Parameters", f"{total_params:,}"],
            ["", ""],
            ["Test Loss", f"{test_loss:.4f}"],
            ["Test Accuracy", f"{test_acc:.4f}"],
            ["Accuracy", f"{acc:.4f}"],
            ["Macro F1", f"{f1_m:.4f}"],
            ["Weighted F1", f"{f1_w:.4f}"],
        ]

        table = plt.table(
            cellText=summary_table,
            colLabels=["Metric", "Value"],
            colLoc="left",
            cellLoc="left",
            loc="center"
        )
        table.scale(1, 1.6)
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        pdf.savefig(fig)
        plt.close()

        # ---------- PAGE 2: LOSS CURVE ----------
        fig = plt.figure(figsize=(8, 6))
        plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
        plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        pdf.savefig(fig)
        plt.close()

        # ---------- PAGE 3: ACCURACY CURVE ----------
        fig = plt.figure(figsize=(8, 6))
        plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
        plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        pdf.savefig(fig)
        plt.close()

        # ---------- PAGE 4: CONFUSION MATRIX ----------
        fig = plt.figure(figsize=(8, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        pdf.savefig(fig)
        plt.close()

        # ---------- PAGE 5: PER-CLASS F1 ----------
        fig = plt.figure(figsize=(9, 5))
        f1_scores = [class_report[c]["f1-score"] for c in class_labels]
        plt.bar(class_labels, f1_scores)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1 Score")
        plt.title("Per-Class F1 Score")
        plt.grid(axis="y", alpha=0.3)
        pdf.savefig(fig)
        plt.close()

        # ---------- PAGE 6: CONFIDENCE HISTOGRAM ----------
        fig = plt.figure(figsize=(8, 5))
        plt.hist(confidence, bins=30)
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Count")
        plt.title("Prediction Confidence Distribution")
        plt.grid(alpha=0.3)
        pdf.savefig(fig)
        plt.close()

    print(f"PDF report saved as: {pdf_name}")

    # =========================
    # RETURN EVERYTHING (IMPORTANT)
    # =========================
    return {
        "metrics": {
            "test_loss": test_loss,
            "accuracy": acc,
            "macro_f1": f1_m,
            "weighted_f1": f1_w,
            "macro_precision": precision_m,
            "macro_recall": recall_m,
            "weighted_precision": precision_w,
            "weighted_recall": recall_w,
        },
        "per_class_report": class_report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
        "confidence": confidence,
        "metadata": {
            "dataset": dataset_name,
            "model": model_name,
            "img_size": img_size,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "class_indices": test_it.class_indices,
            "class_weights": class_weights,
            "training_time_seconds": training_time_seconds,
        }
    }
