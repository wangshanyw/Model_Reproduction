import tensorflow as tf
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

# add one more dimension
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# create train dataset & test dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(100)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(100)


# create the LeNetModel
class LeNetModel(tf.keras.Model):
    def __init__(self):
        super(LeNetModel, self).__init__()
        # 1st convolution and subsampling layers
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(5, 5), 
                     padding="SAME", activation='relu', use_bias=True, 
                     bias_initializer='zeros')
        self.maxpool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                        padding='same')
        # 2nd convolution and subsampling layers
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(5, 5), 
                     strides=(1, 1), padding="SAME", activation='relu',  
                     use_bias=True, bias_initializer='zeros')
        self.maxpool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                        padding='same')
        # the flattening layer
        self.flatten = layers.Flatten()
        # 1st full connection layer
        self.full1 = layers.Dense(units=512, activation='relu', 
                     use_bias=True, bias_initializer='zeros')
        # 2nd full connection layer
        self.full2 = layers.Dense(units=10, activation='softmax', 
                     use_bias=True, bias_initializer='zeros')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.full1(x)
        x = self.full2(x)
        return x

model = LeNetModel()

# define how to calculate loss and accuracy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy  (name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy  (name='test_accuracy')

# define the step of training
@tf.function
def train_step(image, labels):
    with tf.GradientTape() as tape: # GradientTape梯度带
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # apply_gradients 作用：把计算出来的梯度更新到变量上面去。
    train_loss(loss)
    train_accuracy(labels, predictions)


# define the step of testing
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_fn(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


# start to train & test
EPOCHS = 40
for epoch in range(EPOCHS):
    for images, labels in train_dataset:
        train_step(images, labels)
    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, '\
               'Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
