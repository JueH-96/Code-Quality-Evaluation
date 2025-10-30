# YOUR CODE HERE
import sys
from collections import Counter

def mex(A):
    A = sorted(list(set(A)))
    for i in range(len(A)):
        if A[i] != i:
            return i
    return len(A)

N, Q = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
A_counter = Counter(A)
queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

for query in queries:
    i, x = query
    i -= 1
    A_counter[A[i]] -= 1
    if A_counter[A[i]] == 0:
        del A_counter[A[i]]
    A[i] = x
    if x not in A_counter:
        A_counter[x] = 0
    A_counter[x] += 1
    print(mex(A_counter))