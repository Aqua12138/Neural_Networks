import tensorflow as tf
import os
import urllib.request as load #download the file
import zipfile
import random
import shutil
from shutil import copyfile
class Model():
    def __init__(self):
        self.data_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
        self.data_file_name = "catsdogs.zip"
        self.download_dir = '/home/zhx/tensorflow_model/data/'
    def zipfile(self):
        if os.path.exists('catsdogs.zip'):
            print('it has been downed')
        else:
            load.urlretrieve(self.data_url,self.data_file_name)
            zip_file = zipfile.ZipFile(self.data_file_name)
            zip_file.extractall(self.download_dir)
            zip_file.close()
            print("Number of cat images:", len(os.listdir('/home/zhx/tensorflow_model/data/PetImages/Cat/')))
    def mkdir(self):
        try:
            #if it exist,it won't create
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs')
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs/training')
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs/testing')
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs/training/cats')
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs/training/dogs')
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs/testing/cats')
            os.mkdir('/home/zhx/tensorflow_model/data/cats-v-dogs/testing/dogs')
        except OSError:
            pass
    def splite_data(self,source,training,testing,split_size):
        #traverse the all file
        files = []
        for filename in os.listdir(source):
            file = source+filename
            if os.path.getsize(file)>0:
                files.append(filename)
            else:
                print(filename +'is zero length,so ignore')
        training_length = int(len(files)*split_size)
        testing_length = int(len(file)-training_length)
        shuffled_set = random.sample(files, len(files))#extract the all
        training_set = shuffled_set[:training_length]
        testing_set = shuffled_set[training_length:]
        for filename in training_set:
            this_file = source + filename
            destination = training + filename
            copyfile(this_file,destination)

        for filename in testing_set:
            this_file = source + filename
            destination = testing + filename
            copyfile(this_file, destination)
    def one_time_getdata(self):
        M = Model()
        M.zipfile()
        M.mkdir()

        CAT_SOURCE_DIR = "/home/zhx/tensorflow_model/data/PetImages/Cat/"
        TRAINING_CATS_DIR = "/home/zhx/tensorflow_model/data/cats-v-dogs/training/cats/"
        TESTING_CATS_DIR = "/home/zhx/tensorflow_model/data/cats-v-dogs/testing/cats/"
        DOG_SOURCE_DIR = "/home/zhx/tensorflow_model/data/PetImages/Dog/"
        TRAINING_DOGS_DIR = "/home/zhx/tensorflow_model/data/cats-v-dogs/training/dogs/"
        TESTING_DOGS_DIR = "/home/zhx/tensorflow_model/data/cats-v-dogs/testing/dogs/"

        split_size = 0.9
        M.splite_data(CAT_SOURCE_DIR,TRAINING_CATS_DIR,TESTING_CATS_DIR,split_size)
        M.splite_data(DOG_SOURCE_DIR,TRAINING_DOGS_DIR,TESTING_DOGS_DIR,split_size)

        print("Number of training cat images", len(os.listdir('/home/zhx/tensorflow_model/data/cats-v-dogs/training/cats/')))
        print("Number of training dog images", len(os.listdir('/home/zhx/tensorflow_model/data/cats-v-dogs/training/dogs/')))
        print("Number of testing cat images", len(os.listdir('/home/zhx/tensorflow_model/data/cats-v-dogs/testing/cats/')))
        print("Number of testing dog images", len(os.listdir('/home/zhx/tensorflow_model/data/cats-v-dogs/testing/dogs/')))

M = Model()
#M.one_time_getdata()#first to do,and do not repeat the operator