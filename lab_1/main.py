import numpy
from mpi4py import MPI

COLS = int(100)
ROWS = int(100)
generations = 1000

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
stat = MPI.Status()

if size > ROWS:
    print("Not enough ROWS")
    exit()

subROWS = ROWS//size+2
if size == 1:
    subROWS = ROWS

def msgUp(subGrid):
        comm.send(subGrid[subROWS-2,:],dest=rank+1)
        subGrid[subROWS-1,:]=comm.recv(source=rank+1)
        return 0

def msgDn(subGrid):
        comm.send(subGrid[1,:],dest=rank-1)
        subGrid[0,:] = comm.recv(source=rank-1)
        return 0

def newGenetation(subGrid):
    intermediateG = numpy.copy(subGrid)
    for ROWelem in range(1,subROWS-1):
        for COLelem in range(1,COLS-1):
            neighbour_sum = ( subGrid[ROWelem-1,COLelem-1]+subGrid[ROWelem-1,COLelem]+subGrid[ROWelem-1,COLelem+1]
                            +subGrid[ROWelem,COLelem-1]+subGrid[ROWelem,COLelem+1]
                            +subGrid[ROWelem+1,COLelem-1]+subGrid[ROWelem+1,COLelem]+subGrid[ROWelem+1,COLelem+1] )
            if subGrid[ROWelem,COLelem] == 1:
                if neighbour_sum < 2:
                    intermediateG[ROWelem,COLelem] = 0
                elif neighbour_sum > 3:
                    intermediateG[ROWelem,COLelem] = 0
                else:
                    intermediateG[ROWelem,COLelem] = 1
            if subGrid[ROWelem,COLelem] == 0:
                if neighbour_sum == 3:
                    intermediateG[ROWelem,COLelem] = 1
                else:
                    intermediateG[ROWelem,COLelem] = 0
    subGrid = numpy.copy(intermediateG)
    return subGrid



if __name__=="__main__":
    start_time = MPI.Wtime()

    N = numpy.random.binomial(1,0.2,size=(subROWS+2)*COLS)
    subGrid = numpy.reshape(N,(subROWS+2,COLS))
    subGrid[:,0] = 0
    subGrid[:,-1] = 0


    if rank == 0:
        subGrid[0,:] = 1

    result = None
    oldGrid=comm.gather(subGrid[1:subROWS-1,:],root=0)
    for i in range(1, generations):
        subGrid = newGenetation(subGrid)
        if size == 1:
            continue
        if rank == 0:
            msgUp(subGrid)
        elif rank == size-1:
            msgDn(subGrid)
        else:
            msgUp(subGrid)
            msgDn(subGrid)
        newGrid=comm.gather(subGrid[1:subROWS-1,:],root=0)
        if rank == 0: 
            result = numpy.vstack(newGrid)
    end_time = MPI.Wtime() - start_time
    result = comm.reduce(end_time, op=MPI.MAX, root=0)
    if rank == 0:
        print(f",{result}")
    