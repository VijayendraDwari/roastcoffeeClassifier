"""Original CNN topology, with explicit RGB inputs and train-only augmentation."""
import tensorflow as tf


def build_model(size=50, l2_strength=0.02, learning_rate=0.0005, seed=42):
    L = tf.keras.layers
    model = tf.keras.Sequential([L.Input((size,size,3)), L.RandomFlip('horizontal', seed=seed), L.RandomRotation(0.0556, seed=seed+1), L.RandomTranslation(0.2,0.2,seed=seed+2), L.RandomZoom(0.2,seed=seed+3)])
    for filters in [32,64,64]:
        model.add(L.Conv2D(filters,3,kernel_regularizer=tf.keras.regularizers.l2(l2_strength)))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
        model.add(L.MaxPooling2D(2))
    model.add(L.Dropout(0.25,seed=seed+4))
    model.add(L.Flatten())
    for _ in range(2):
        model.add(L.Dense(128,kernel_regularizer=tf.keras.regularizers.l2(l2_strength)))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
    model.add(L.Dense(4,activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
