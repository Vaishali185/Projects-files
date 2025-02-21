
Training of PyTorch network over MPI:
Training of PyTorch network over MPI is executed in the file scalable_analytics1.py and it is executed on the anaconda command prompt using following command.
mpiexec -n 4 python -m mpi4py scalable_analytics1.py


Implementing a pipelined prediction:
Pipelined prediction is implemented in the file scalable_analytics2.py and it is executed on the anaconda command prompt using following command.
mpiexec -n 2 python -m mpi4py scalable_analytics2.py

Pipelined prediction is also implemented in scalable_analytics3.py file in which node 0 loads all images and produce pillow images. node 1 resize them and transform them into NumPy arrays. node 2 will be used for prediction. It is executed on the anaconda command prompt using following command.
mpiexec -n 3 python -m mpi4py scalable_analytics3.py


scalable_analytics5.py file implements the task in which node 0 scatters the images over other nodes. node 0 load all the images and produce Pillow images, Node 1 could resize them and transform them into NumPy arrays. node 2 and 3 performs prediction task. This is executed as follows:
mpiexec -n 3 python -m mpi4py scalable_analytics5.py

Output of this execution is shown in the ipynb file attached. This ipynb file also contains all this codes ,its output and explaination of code. 