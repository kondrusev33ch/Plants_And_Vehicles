import matplotlib.pyplot as plt


def plot_training_curves(train_history):
    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_batch_images(generator, class_map):
    images, labels = next(iter(generator))

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()
    for image, label, ax in zip(images[:8], labels[:8], axes):  # ограничим batch до 8 изображений
        ax.set_title(class_map[label.argmax()])
        ax.imshow(image)
    plt.show()

if __name__ == '__main__':
    pass
