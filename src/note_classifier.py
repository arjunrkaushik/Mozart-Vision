import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.metrics import Precision, Recall, SparseCategoricalAccuracy
from keras.models import Sequential


def train_classifier():
    dataset_path = "../Data/Notes"

    data = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, color_mode='grayscale')

    class_names = data.class_names

    data = data.map(lambda x, y: (x / 255, y))
    scaled_iterator = data.as_numpy_iterator()

    no_of_batches = len(data)
    train_size = int(no_of_batches * 0.7)
    val_size = int(no_of_batches * 0.2) + 1
    test_size = int(no_of_batches * 0.1) + 1

    # print(no_of_batches, train_size+val_size+test_size)

    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size + val_size).take(test_size)

    plt.figure(figsize=(10, 10))

    for images, labels in train_data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            temp = tf.squeeze(images[i].numpy().astype("uint8"))
            plt.imshow(temp, cmap=plt.cm.binary)
            plt.title(class_names[labels[i]])
            plt.axis("off")

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_normal',
                     input_shape=(256, 256, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.fit(train_data,
              epochs=3,
              callbacks=[callback],
              verbose=1,
              validation_data=val_data)

    # Metrics

    precision, recall, acc = Precision(), Recall(), SparseCategoricalAccuracy()

    test_iterator = test_data.as_numpy_iterator()
    for batch in test_iterator:
        X, y = batch
        pred = model.predict(X)
        precision.update_state(y, pred.argmax(axis=1))
        recall.update_state(y, pred.argmax(axis=1))
        acc.update_state(y, pred)

    print("Precision:", precision.result().numpy)
    print("Recall: ", recall.result().numpy)
    print("Accuracy:", acc.result().numpy)

    model.save('../model/note_classifier_model.h5')
