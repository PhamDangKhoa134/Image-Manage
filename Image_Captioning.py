import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.preprocessing.text import tokenizer_from_json

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
attention_features_shape = 64
embedding_dim = 256
units = 512
vocab_size = 5001
max_length=31
IMAGE_SHAPE = (299, 299)

# Đọc lại tokenizer từ file JSON
with open("tokenizer_tieng_viett.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

def preprocess_images_function(image_path):
    img = tf.io.read_file(image_path, name=None)
    img = tf.image.decode_jpeg(img, channels=0)
    img = tf.image.resize(img, IMAGE_SHAPE)
    img = tf.keras.applications.inception_v3.preprocess_input(img, data_format=None) # CHuẩn hóa trong khoảng từ -1 -> 1
    return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

new_input = image_model.input  # Lấy đối tượng đầu vào của image_model
hidden_layer = image_model.layers[-1].output  # Lấy đối tượng đầu ra của image_model

image_features_extract_model = tf.compat.v1.keras.Model(new_input, hidden_layer)
# Nén dữ liệu nhưng vẫn giữ các đặc trưng
class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)
    def call(self, features):
        features = self.dense(features)
        features = tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0)
        return features
encoder=Encoder(embedding_dim)
# Lọc ra các đặc trưng theo trọng số
class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 =  tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units=units

    def call(self, features, hidden):
        hidden_with_time_axis=hidden[:, tf.newaxis]
        score =tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights =  tf.keras.activations.softmax(self.V(score), axis=1)
        context_vector =  attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
# Trả lại về dạng dữ liệu gốc
class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units)
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units)
        self.d2 = tf.keras.layers.Dense(vocab_size)


    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        embed = self.embed(x)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        output,state = self.gru(embed)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.d2(output)

        return output,state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
decoder=Decoder(embedding_dim, units, vocab_size)
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.init_state(batch_size=1)

    temp_input = tf.expand_dims(preprocess_images_function(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['start']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id =  tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == 'end':
            return result, attention_plot,predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot,predictions
# Khôi phục các mô hình và trọng số từ checkpoint
checkpoint_path = "checkpoint_tieng_viet"
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Khôi phục mô hình từ checkpoint mới nhất
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print("Checkpoint restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("No checkpoint found")

# Dự đoán 1 ảnh
test_image = "IMG_4538.JPG"
result, attention_plot,pred_test = evaluate(test_image)
pred_caption = ' '.join([word for word in result if word != "end"]).rsplit(' ', 0)[0]
print("Prediction Caption:", pred_caption)

# Dự đoán nhiều ảnh
link_image_file = "C:/TaiLieu/Deep_Learning/Image_Manage/link_imgs.txt"
pred_captions_file = "C:/TaiLieu/Deep_Learning/Image_Manage/pred_captions.txt"

with open(link_image_file, "r") as file:
    image_paths = file.read().splitlines()

with open(pred_captions_file, "w", encoding="utf-8") as output_file:
    for test_image in image_paths:
        try:
            result, attention_plot, pred_test = evaluate(test_image)
            pred_caption = ' '.join([word for word in result if word != "end"]).rsplit(' ', 0)[0]
            print(f"Image: {test_image} -> Prediction Caption: {pred_caption}")
            output_file.write(f"{pred_caption}\n")
        except Exception as e:
            print(f"Error with image {test_image}: {e}")
            output_file.write(f"{test_image}: Error occurred\n")

while True:
    # Nhập đường dẫn ảnh từ người dùng
    test_image = input("Nhập đường dẫn ảnh (hoặc nhập 'q' để thoát): ").strip()
    
    if test_image.lower() == 'q':
        print("Chương trình kết thúc.")
        break  # Thoát khỏi vòng lặp nếu người dùng nhập 'q'

    try:
        result, attention_plot, pred_test = evaluate(test_image)
        pred_caption = ' '.join([word for word in result if word != "end"]).rsplit(' ', 0)[0]
        print(f"Ảnh: {test_image} -> Caption dự đoán: {pred_caption}")
    except Exception as e:
        print(f"Lỗi với ảnh {test_image}: {e}")
        



