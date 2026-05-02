from pymongo import MongoClient
import os
MONGO_URI = "mongodb+srv://medicalrecord:123@cluster0.hqwl0gl.mongodb.net/medical_records?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

db = client["datasets"] 
patient_collection = db["dataset"] 
