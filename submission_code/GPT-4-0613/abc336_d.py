import sys
from collections import deque
from heapq import heapify, heappop, heappush

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = deque([])
    C = []
    heapify(C)
    for i in range(N):
        if i % 2 == 0:
            B.append(A[i])
        else:
            B.appendleft(A[i])
    for i in range(N):
        if i % 2 == 0:
            heappush(C, B[i])
        else:
            if C[0] < B[i]:
                heappop(C)
                heappush(C, B[i])
    print(len(C))

solve()