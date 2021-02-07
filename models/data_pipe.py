import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Data_pipeline():
    """Load and split data into training and test set according 
    to the parameters chosen 
    """    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dtype = self.kwargs['dtype']
        self.len = self.kwargs['len']
         # number of samples generated
        self.num_phase_1 = self.kwargs['num_phase_1']
        self.num_phase_2 = self.kwargs['num_phase_2']
        # interaction strengths
        self.j1 = self.kwargs['j1']
        self.j2 = self.kwargs['j2']
        #print(str(float(self.j1)), str(int(self.len)))
        self.p1 = self.load_data(num = self.num_phase_1, j = self.j1)
        self.p2 = self.load_data(num = self.num_phase_2, j = self.j2)
    
        
    def load_data(self, num, j):
        """load the data by specifying the data type, 
        interaction strengths and number of samples.
        Note This is only useful if you run the en_spectrum.py 
        script to generate your own data set.
        for entanglement specta, the size is determined by the cut position
        for the wavefunction, it's by the system size
        """        
        loaded = np.load('data_set/j_'+str(j)+'/1_en_spectrum_L='
                        +str(self.len)+'_j='+str(j)+'.npz')        
        dim = loaded[self.dtype].shape
        x = np.zeros((num, dim[0], dim[1]), dtype='float64')
        # loop through files to load data
        for i in range(num):
            filename = "data_set/j_"+str(j)+"/"+str(i+1)+"_en_spectrum_L="+str(self.len)+"_j="+str(j)+".npz"
            loaded = np.load(filename)
            if self.dtype == 'wave':
                wav = loaded[self.dtype]
                x[i] = np.multiply(wav, np.conj(wav)).real
            else:
                x[i] = loaded[self.dtype]
        
        phase = np.concatenate((x[:])).astype('float32')
        return phase
    
    def split_data(self, test_size = 0.3):
        """Train-test split

        Keyword Arguments:
            test_size {float} -- The portion of test set (default: {0.3})

        Returns:
            X_train, X_test, Y_train, Y_test
        """        
        # create labels for the two phases
        label_1 = np.ones(self.p1.shape[0],dtype=np.int8)
        label_2 = np.zeros(self.p2.shape[0],dtype=np.int8)
        # combine labels and the data
        phase_1 = np.column_stack((label_1, self.p1))
        phase_2 = np.column_stack((label_2, self.p2))
        # train_test split
        dat = np.concatenate((phase_1, phase_2))
        X_train, X_test, Y_train, Y_test = train_test_split(dat[:,1:], dat[:,:1], test_size=test_size, random_state=1)
        # convert class vectors to binary class matrices
        enc = OneHotEncoder(dtype='float32')
        Y_train = enc.fit_transform(Y_train)
        Y_test = enc.transform(Y_test)
    
        return X_train, X_test, Y_train, Y_test