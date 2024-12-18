from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from bson.objectid import ObjectId
from io import BytesIO
import datetime
from flask_cors import CORS
import base64
from bson import Binary
from PIL import Image

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.preprocessing.text import tokenizer_from_json
import os
from rank_bm25 import BM25Okapi

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
    img = tf.keras.applications.inception_v3.preprocess_input(img, data_format=None) 
    return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

new_input = image_model.input  
hidden_layer = image_model.layers[-1].output  

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
checkpoint_path = "checkpoint_tieng_viet_moi"
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Khôi phục mô hình từ checkpoint mới nhất
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print("Checkpoint restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("No checkpoint found")


app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
collection = db["images"]

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    location = request.form.get('location', 'Unknown')
    timestamp = request.form.get('timestamp', datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"))

    try:
        image = Image.open(uploaded_file)
        image = image.resize((299, 299))

        if not os.path.exists('static'):
            os.makedirs('static')

        image_path = os.path.join('static', filename)
        image.save(image_path)

        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        binary_data = Binary(base64.b64decode(encoded_string))

        # Giả sử hàm evaluate() trả về caption cho ảnh
        result, attention_plot, pred_test = evaluate(image_path)
        caption = ' '.join([word for word in result if word != "end"]).rsplit(' ', 0)[0]
        # caption = ""

        image_id = collection.insert_one({
            "filename": filename,
            "data": binary_data,
            "location": location,
            "caption": caption,
            "timestamp": timestamp
        }).inserted_id

        # Lấy tất cả các caption từ MongoDB để áp dụng BM25
        all_documents = list(collection.find({}, {"caption": 1, "_id": 0}))
        corpus = [doc["caption"].split() for doc in all_documents]

        # Khởi tạo BM25 và tính điểm cho các caption
        bm25 = BM25Okapi(corpus)
        query = caption.split()
        scores = bm25.get_scores(query)

        # Sắp xếp các caption theo điểm BM25
        ranked_captions = sorted(
            zip(all_documents, scores), key=lambda x: x[1], reverse=True
        )

        # Trả về top 5 caption tương tự
        top_results = [{"caption": doc["caption"], "score": score} for doc, score in ranked_captions[:5]]

        return jsonify({
            "message": "Image uploaded successfully",
            "image_id": str(image_id),
            "filename": filename,
            "location": location,
            "timestamp": timestamp,
            "caption": caption,
            "similar_captions": top_results
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_image/<image_id>', methods=['GET'])
def get_image(image_id):
    try:
        image_doc = collection.find_one({"_id": ObjectId(image_id)})
        if not image_doc:
            return jsonify({"error": "Image not found"}), 404

        # Chuyển dữ liệu ảnh thành BytesIO để trả về qua API
        image_data = BytesIO(image_doc['data'])
        return send_file(image_data, mimetype='image/jpeg', download_name=image_doc['filename'])
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_image_info/<image_id>', methods=['GET'])
def get_image_info(image_id):
    try:
        image_doc = collection.find_one({"_id": ObjectId(image_id)})
        if not image_doc:
            return jsonify({"error": "Image not found"}), 404

        return jsonify({
            "filename": image_doc["filename"],
            "location": image_doc["location"],
            "caption": image_doc["caption"],
            "timestamp": image_doc["timestamp"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/search_and_get_info/<text>', methods=['GET'])
def search_and_get_info(text):
    try:
        # Tìm các tài liệu theo từ khóa text
        result = collection.find({'$text': {'$search': text}})
        
        # Duyệt qua từng tài liệu và lấy thông tin chi tiết
        images_info = []
        for doc in result:
            image_info = {
                "image_id": str(doc["_id"]),
                "filename": doc.get("filename", ""),
                "location": doc.get("location", ""),
                "caption": doc.get("caption", ""),
                "timestamp": doc.get("timestamp", "")
            }
            images_info.append(image_info)

        # Kiểm tra nếu không có kết quả
        if not images_info:
            return jsonify({"error": "No images found"}), 404

        # Trả về danh sách thông tin chi tiết của các ảnh
        return jsonify({"images": images_info}), 200

    except Exception as e:
        # Xử lý ngoại lệ và trả về thông báo lỗi
        return jsonify({"error": str(e)}), 500


@app.route('/search_images', methods=['GET'])
def search_images():
    try:
        # Nhận các tham số từ yêu cầu GET
        start_date = request.args.get('start_date')  
        end_date = request.args.get('end_date')      
        location = request.args.get('location', '')  
        keyword = request.args.get('keyword', '')    

        # Tạo bộ lọc cho khoảng thời gian
        query = {}
        if start_date and end_date:
            query["timestamp"] = {
                "$gte": start_date,
                "$lte": end_date
            }

        if location:
            query["location"] = location

        result = collection.find(query)

        if keyword:
            filtered_result = [
                doc for doc in result if keyword.lower() in doc["caption"].lower()
            ]
        else:
            filtered_result = list(result)

        if not filtered_result:
            return jsonify({"error": "No images found"}), 404

        # Chuẩn bị danh sách thông tin để trả về
        images_info = []
        for doc in filtered_result:
            image_info = {
                "image_id": str(doc["_id"]),
                "filename": doc.get("filename", ""),
                "location": doc.get("location", ""),
                "caption": doc.get("caption", ""),
                "timestamp": doc.get("timestamp", "")
            }
            images_info.append(image_info)

        # Trả về danh sách thông tin của các ảnh tìm được
        return jsonify({"images": images_info}), 200

    except Exception as e:
        # Xử lý ngoại lệ và trả về thông báo lỗi
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

