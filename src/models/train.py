import torch


def train_student(model, train_loader, val_loader, criterion, optimizer, config, teacher, DEVICE="cpu"):
    loss_hist = {"train accuracy": [], "train loss": [], "val accuracy": []}

    teacher.to(DEVICE)

    for epoch in range(1, config.NUM_EPOCHES + 1):
        model.train()

        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        for batch_idx, (img, labels) in enumerate(train_loader):
            img = img.to(DEVICE)
            labels = labels.to(DEVICE)

            #preds, teacher_preds = model(img)
            preds = model(img)
            teacher_preds = teacher(img)

            #print(teacher_preds.size())
            #print(type(preds))

            loss = criterion(teacher_preds, preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())

            epoch_train_loss += loss.item()

        with torch.no_grad():
            model.eval()

            y_true_test = []
            y_pred_test = []

            for batch_idx, (img, labels) in enumerate(val_loader):
                img = img.to(DEVICE)
                labels = labels.to(DEVICE)

                #preds, teacher_preds = model(img)
                preds = model(img)

                y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
                y_true_test.extend(labels.detach().tolist())

                test_total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x == y])
                test_total = len(y_pred_test)
                test_accuracy = test_total_correct * 100 / test_total

        loss_hist["train loss"].append(epoch_train_loss)

        total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x == y])
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total

        loss_hist["train accuracy"].append(accuracy)
        loss_hist["val accuracy"].append(test_accuracy)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("       Validation Accuracy%: ", test_accuracy, "==", test_total_correct, "/", test_total)
        print("-------------------------------------------------")

    return loss_hist
