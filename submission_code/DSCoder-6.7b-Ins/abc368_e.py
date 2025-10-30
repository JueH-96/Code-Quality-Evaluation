import sys
from operator import itemgetter
from heapq import *

input = sys.stdin.readline

def solve():
    N, M, X1 = map(int, input().split())
    trains = [list(map(int, input().split())) for _ in range(M)]

    trains.sort(key=itemgetter(2, 3))
    trains.append([0, 0, 0, 0])

    X = [0]*M
    heap = []

    for i in range(M+1):
        while heap and heap[0][0] < trains[i][2]:
            heappop(heap)
        if i:
            X[i-1] = -heap[0][0] + trains[i][2] - X1
        if heap:
            heappush(heap, (-(heap[0][0] - trains[i][2] + X[i-1] + X1)))
        heappush(heap, (-trains[i][3]))

    print(' '.join(map(str, X)))

solve()