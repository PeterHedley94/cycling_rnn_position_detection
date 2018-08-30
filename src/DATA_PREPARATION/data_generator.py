import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')

from src.DATA_PREPARATION.folder_manipulation import *
#from src.PREPROCESSING.preprocessing import *
#from keras.utils import to_categorical
from PIL import Image
import json
import cv2




# Class to yield batches of image data indefinitely for as many epochs as required
class DataGenerator(object):
    def __init__(self,dir, batch_size = 16, shuffle = True, sequence_length = 1,time_gap=1,time_distributed = False, debug_mode = False,images = False,
                                                                    image_height = 0,image_width=0,no_image_channels=0,camera_model_location = os.path.join("utils",'camera_model.json')):
        print("New instance of 'datagenerator' initialised on {}".format(dir))
        # Attributes initialised directly from arguments

        self.image_height = image_height
        self.image_width = image_width
        self.no_image_channels = no_image_channels
        self.time_gap = time_gap
        self.n_outputs = 3 #dx,dy,dz
        self.data_dims = (3,4) #3x4 camera matrix
        self.data_length = 16 #3x4 + ecz
        self.time_distributed_ = time_distributed
        self.debug_mode_ = debug_mode
        self.use_images = images
        if self.use_images:
            print("Images enabled!!")

        self.shuffle = shuffle
        self.sequence_length_ = sequence_length

        self.epoch_number = 1
        with open(camera_model_location) as f:
            self.camera_model = json.load(f)

        #Central point matrix [c1,c2]
        self.c_mat = np.zeros((3,1))
        self.c_mat[1,:],self.c_mat[0,:] = self.camera_model["principal_point"]
        #Focal Matrix [[1/f1,0],[0,1/f2]]
        self.foc_mat = np.zeros((2,2))
        self.foc_mat[0,0],self.foc_mat[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]
        #Focal Matrix Inv [[1/f1,0],[0,1/f2]]
        self.foc_mat_inv = np.zeros((2,2))
        self.foc_mat_inv[0,0],self.foc_mat_inv[1,1] = self.camera_model["focal_length"][0],self.camera_model["focal_length"][1]
        #Camera Transformed
        self.T_SC = np.array(self.camera_model["T_SC"]).reshape((4,4))

        # Produce lists of all the image sequence files in the folders (trainx)
        # and corresponding one-hot array labels (trainy)
        self.file_numbers,self.T_WS_C,self.T_WS_r,self.sb,self.images = self.get_files(dir)

        # Batch size is defined here as the size of the first dimension of the output
        # If not time distributed this is the number of images in the batch
        # If time distributed there will be batch_size*sequence_length images per batch
        self.batch_size = batch_size
        self.effective_batch_size = batch_size
        self.index_buffer = self.sequence_length_
        #CHENGED TO if ... instead of if not ...?
        # Set the number of batches per epoch to iterate over in generation
        self.batches_per_epoch =  int( np.floor( ( len(self.file_numbers) / self.batch_size ) -self.index_buffer ) )
        while self.batches_per_epoch < 1 and self.batch_size >1:
            self.batch_size -= 1
            self.effective_batch_size -= 1
            self.batches_per_epoch =  int( np.floor( ( len(self.file_numbers) / self.batch_size ) -self.index_buffer ) )


    def generate(self):
        # For as many epochs as required:
        while 1:
            #dont want sequence near end
            indexes = self.__get_exploration_order(self.file_numbers[:-self.index_buffer])
            # Generate batches
            for i in (range(self.batches_per_epoch)):
                batch_indexes = indexes[i*self.effective_batch_size:(i+1)*self.effective_batch_size]
                yield(self.__data_generation(batch_indexes))
            self.epoch_number+=1

    def gd(self,filename):
        return np.loadtxt(filename)

    def to_number(self,item):
        item = item.split("/")[-1]
        number = int(item.split("_")[0])
        return number

    def world_to_camera(self,array):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(self.T_SW,world_point)
        camera_point[:3,:] = camera_point[:3,:]/camera_point[2,:]
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point[:3,:]

    def camera_to_world(self,array):
        #array is (3,N)
        N = array.shape[1]
        camera_point = np.ones((4,N))
        camera_point[:3,:] = array
        camera_point[:3,:] = camera_point[:3,:]-self.c_mat
        camera_point[:2,:] = np.matmul(self.foc_mat,camera_point[:2,:])
        camera_point[:2,:] = camera_point[:2,:]*camera_point[2,:]
        world_point = np.matmul(self.T_WS,camera_point)[:3,:]
        return world_point


    #Get lists of the image folders and labels
    def get_files(self,dir):
        data_folder = os.path.join(dir,"pose")
        print("Searching in directory " + str(dir))
        o_T_WS_C = get_file_names(data_folder,"T_WS_C.txt",self.to_number)[::self.time_gap]
        o_T_WS_r = get_file_names(data_folder,"T_WS_r.txt",self.to_number)[::self.time_gap]
        o_sb = get_file_names(data_folder,"sb.txt",self.to_number)[::self.time_gap]
        file_numbers = np.arange(len(o_T_WS_C))
        images = get_image_names(os.path.join(dir,"cam0","data"))[::self.time_gap]
        return file_numbers,o_T_WS_C,o_T_WS_r,o_sb,images

    def __get_exploration_order(self, list_IDs):
        indexes = np.arange(len(list_IDs)-1)+1 #dont use first one
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def get_T_WC(self,T_WS_C,T_WS_r):
        T_WS = np.zeros((4,4),dtype = np.float64)
        T_WS[:3,:4] = np.concatenate((T_WS_C,T_WS_r[:,None]),axis = 1)
        T_WS[3,3] = 1
        T_WC = np.matmul(T_WS,self.T_SC)
        return T_WC

    def world_to_camera(self,array,T_CW):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(T_CW,world_point)
        camera_point[:2,:] = camera_point[:2,:]/camera_point[2,:]
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point[:3,:]

    def transform_TW(self,TW_to,TW_from):
        return np.matmul(TW_to,TW_from)


    def reverse_transform(self,array):
        reversed = np.zeros((4,4),dtype = np.float64)
        reversed[:3,:4] = np.concatenate((array[:3,:3].transpose(),
                                                np.matmul(-array[:3,:3].transpose(),array[:3,3:])),axis = 1)
        reversed[3,3] = 1
        return reversed

    # Get the data and labels for a batch of folders
    def __data_generation(self,batch_indexes):

        X = None
        X = np.zeros(shape=(self.batch_size, self.sequence_length_,self.data_length))
        Y = np.zeros(shape=(self.batch_size, self.n_outputs))

        if self.debug_mode_:
            X_initial_matrices = np.zeros(shape=(self.batch_size,self.data_dims[0],self.data_dims[1]))

        if self.use_images:
            #####################
            if self.debug_mode_:
                X_images = np.zeros(shape=(self.batch_size, self.sequence_length_,self.image_height,self.image_width,self.no_image_channels),dtype = np.uint8)
            else:
                X_images = np.zeros(shape=(self.batch_size, self.sequence_length_,self.image_height,self.image_width,self.no_image_channels),dtype = np.float32)
        # Loop over the input: make call to dstack then append result with label:
        k = 0
        for seq_no in range(self.effective_batch_size):

            index = batch_indexes[seq_no]
            #Get point from before
            T_WC_initial = self.get_T_WC(self.gd(self.T_WS_C[index-1]),self.gd(self.T_WS_r[index-1]))
            T_CW_init = self.reverse_transform(T_WC_initial)

            for no_in_seq in range(0,self.sequence_length_,1):
                T_W_C_next = self.get_T_WC(self.gd(self.T_WS_C[index+no_in_seq]),self.gd(self.T_WS_r[index+no_in_seq]))

                c_ez = self.reverse_transform(T_W_C_next)[:,2].tolist()

                #TODO add c_ez
                T_W_C_transformed = self.transform_TW(T_CW_init,T_W_C_next)[:3,:]
                last_point = T_W_C_transformed[:3,3:]
                T_W_C_transformed = T_W_C_transformed.reshape((1,-1)).tolist()[0]
                #CHANGE TO DEBUG DONT DO THIS + change data length
                T_W_C_transformed.extend(c_ez)
                X[seq_no,no_in_seq,:] = T_W_C_transformed

                if self.debug_mode_ and no_in_seq == self.sequence_length_-1:
                    X_initial_matrices[seq_no,:,:] = T_WC_initial[:3,:]
                #Images
                if self.use_images:
                    resized_image = get_resized_image(self.images[index+no_in_seq],self.image_height,self.image_width)
                    if self.debug_mode_:
                        X_images[seq_no,no_in_seq,:,:,:] = resized_image
                        cv2.imshow('l',resized_image)
                        cv2.waitKey(10)
                    else:
                        X_images[seq_no,no_in_seq,:,:,:] = np.float32(resized_image)

            #Add Y data
            #TODO sort Y data
            y_index = index+self.sequence_length_
            T_W_C_next = self.get_T_WC(self.gd(self.T_WS_C[y_index]),self.gd(self.T_WS_r[y_index]))
            T_W_C_transformed = self.transform_TW(T_CW_init,T_W_C_next)[:3,:]
            Y[seq_no,:,None] = T_W_C_transformed[:3,3:].reshape((-1,1))

        if self.use_images:
            X_images = (X_images - 127.5) / 127.5
            if self.debug_mode_:
                return X_initial_matrices,X_images,X,Y
            else:
                return [X_images,X],Y
        else:
            if self.debug_mode_:
                return X_initial_matrices,X,Y
            return X,Y
