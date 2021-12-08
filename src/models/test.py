import torch


def test(model, test_loader, DEVICE):
    with torch.no_grad():
        model.eval()

        y_true_test = []
        y_pred_test = []

        for batch_idx, (img, labels) in enumerate(test_loader):
            img = img.to(DEVICE)
            labels = labels.to(DEVICE)

            preds, teacher_preds = model(img)
            # preds = model(img)

            y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_test.extend(labels.detach().tolist())

        total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x == y])
        total = len(y_pred_test)
        accuracy = total_correct * 100 / total

        print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)
    return accuracy
