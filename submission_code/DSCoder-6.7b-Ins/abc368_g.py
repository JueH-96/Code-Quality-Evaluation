import sys
from heapq import *

read = sys.stdin.readline

N = int(read())
A = list(map(int, read().split()))
B = list(map(int, read().split()))
Q = int(read())

queries = []
for _ in range(Q):
    queries.append(list(map(int, read().split())))

A_max_heap = []
B_max_heap = []

for i in range(N):
    heappush(A_max_heap, (-A[i], i))
    heappush(B_max_heap, (-B[i], i))

v = 0
answers = []

for query in queries:
    if query[0] == 1:
        _, i, x = query
        _, index = heappushpop(A_max_heap, (-x, i))
        v -= A[index]
        A[index] = x
        heappush(A_max_heap, (-A[index], index))
    elif query[0] == 2:
        _, i, x = query
        _, index = heappushpop(B_max_heap, (-x, i))
        v -= B[index]
        B[index] = x
        heappush(B_max_heap, (-B[index], index))
    else:
        l, r = query[1:]
        l -= 1
        r -= 1
        A_values = [(-A_max_heap[0][0], A_max_heap[0][1])]
        B_values = [(-B_max_heap[0][0], B_max_heap[0][1])]
        while len(A_max_heap) > 1:
            heappop(A_max_heap)
            A_values.append((-A_max_heap[0][0], A_max_heap[0][1]))
        while len(B_max_heap) > 1:
            heappop(B_max_heap)
            B_values.append((-B_max_heap[0][0], B_max_heap[0][1]))
        A_values.sort(key=lambda x: x[1])
        B_values.sort(key=lambda x: x[1])
        A_values = [x[0] for x in A_values]
        B_values = [x[0] for x in B_values]
        for i in range(l, r+1):
            if i % 2 == 0:
                v += A_values[i]
            else:
                v *= B_values[i]
        answers.append(v)
        for value, index in A_values:
            heappush(A_max_heap, (value, index))
        for value, index in B_values:
            heappush(B_max_heap, (value, index))

for answer in answers:
    print(answer)