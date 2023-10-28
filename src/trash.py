import h5py as h5
import numpy as np
import time
import psutil
process = psutil.Process()
memory_info = process.memory_info()
print(f"Utilisation actuelle de la mémoire (en mégaoctets) : {memory_info.rss / (1024 ** 2)}")
File_Name_HDF5='Test.h5'
shape = (10000000, 200)
chunk_shape=(100000, 200)
#Array=np.array(np.random.rand(shape[0]),np.float32)
#
##We are using 4GB of chunk_cache_mem here ("rdcc_nbytes")
#f = h5.File(File_Name_HDF5, 'w',rdcc_nbytes = 1024**2*4000,rdcc_nslots=1e7)
#d = f.create_dataset('Test', shape ,dtype=np.float32,chunks=chunk_shape)
#
##Writing columns
#t1=time.time()
#
#for i in range(0,shape[1]):
#    d[:,i:i+1]=np.expand_dims(Array, 1)
#
#f.close()
#
#print(time.time()-t1)
#
## Reading random rows
## If we read one row there are actually 100 read, but if we access a row
## which is already in cache we would see a huge speed up.
memory_info = process.memory_info()
import os 
print(f"At start Utilisation actuelle de la mémoire (en mégaoctets) : {memory_info.rss / (1024 ** 2)}")
f = h5.File(File_Name_HDF5,'r',rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7)
d = f["Test"]
print(d.shape)
for j in range(0,1000):
    t1=time.time()
    # With more iterations it will be more likely that we hit a already cached row
    size = chunk_shape[0]
    e = j % 10 
    array = d[size * e : size * (e + 1), :]
    #inds=np.random.randint(0, high=shape[0]-1, size=1000)
    #for i in range(0,inds.shape[0]):
    #    Array=np.copy(d[inds[i],:])
    #memory_info = process.memory_info()
    #print(f"Utilisation actuelle de la mémoire (en mégaoctets) : {memory_info.rss / (1024 ** 2)}")
    del array
    print(os.system("free"))
f.close()

