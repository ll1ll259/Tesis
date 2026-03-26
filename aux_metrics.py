import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_score, recall_score, f1_score


def metricas_clasif(trainX, trainY, testX, testY,
                    model=None,
                    train_proba=None,
                    test_proba=None,
                    threshold=0.5):

    """
    Calcula métricas de clasificación para entrenamiento y validación.

    Parameters
    ----------
    trainX, trainY : datos de entrenamiento
    testX, testY : datos de validación
    model : modelo sklearn (opcional)
    train_proba : probabilidades train (si model=None)
    test_proba : probabilidades test (si model=None)
    threshold : punto de corte para predicción
    """

    # Probabilidades
    if model is None:
        proba_train = train_proba
        proba_test = test_proba
    else:
        proba_train = model.predict_proba(trainX)[:,1]
        proba_test = model.predict_proba(testX)[:,1]

    pred_train = (proba_train >= threshold).astype(int)
    pred_test = (proba_test >= threshold).astype(int)

    # ROC AUC
    auc_train = roc_auc_score(trainY, proba_train)
    auc_test = roc_auc_score(testY, proba_test)

    # Gini
    gini_train = 2 * auc_train - 1
    gini_test = 2 * auc_test - 1

    # Accuracy
    acc_train = accuracy_score(trainY, pred_train)
    acc_test = accuracy_score(testY, pred_test)

    # Confusion matrix
    cm_train = confusion_matrix(trainY, pred_train)
    cm_test = confusion_matrix(testY, pred_test)

    # ROC + KS
    fpr_train, tpr_train, _ = roc_curve(trainY, proba_train)
    fpr_test, tpr_test, _ = roc_curve(testY, proba_test)

    ks_train = np.max(tpr_train - fpr_train)
    ks_test = np.max(tpr_test - fpr_test)

    # Precision, Recall, F1
    precision_train = precision_score(trainY, pred_train)
    recall_train = recall_score(trainY, pred_train)
    f1_train = f1_score(trainY, pred_train)

    precision_test = precision_score(testY, pred_test)
    recall_test = recall_score(testY, pred_test)
    f1_test = f1_score(testY, pred_test)

    # Print resumen

    print("\n============== MÉTRICAS DEL MODELO ==============\n")

    print(f"Conteo     Train: {trainY.count():<10,.0f} |    Test: {testY.count():,.0f}")  
    print(f"AUC        Train: {auc_train:<10.4f} |    Test: {auc_test:.4f}")
    print(f"Gini       Train: {gini_train:<10.4f} |    Test: {gini_test:.4f}")
    print(f"KS         Train: {ks_train:<10.4f} |    Test: {ks_test:.4f}")
    print(f"Accuracy   Train: {acc_train:<10.4f} |    Test: {acc_test:.4f}")
    print(f"Precision  Train: {precision_train:<10.4f} |     Test: {precision_test:.4f}")
    print(f"Recall     Train: {recall_train:<10.4f} |     Test: {recall_test:.4f}")
    print(f"F1-Score   Train: {f1_train:<10.4f} |     Test: {f1_test:.4f}")

    # Matriz de confusión
    fig, ax = plt.subplots(1,2, figsize=(8,4))

    ConfusionMatrixDisplay(cm_train).plot(ax=ax[0], cmap="Blues", colorbar=False, values_format=',.0f')
    ax[0].set_title("Confusion Matrix - Train")

    ConfusionMatrixDisplay(cm_test).plot(ax=ax[1], cmap="Blues", colorbar=False, values_format=',.0f')
    ax[1].set_title("Confusion Matrix - Test")

    plt.tight_layout()
    plt.show()

    # ROC curves

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.plot(fpr_train, tpr_train, label=f"AUC = {auc_train:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Train")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(fpr_test, tpr_test, label=f"AUC = {auc_test:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Test")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.tight_layout()
    plt.show()