# YOUR CODE HERE
from sys import stdin

def find_mex(arr):
    seen = set(arr)
    mex = 0
    while mex in seen:
        mex += 1
    return mex

N, Q = map(int, stdin.readline().split())
A = list(map(int, stdin.readline().split()))

for _ in range(Q):
    i, x = map(int, stdin.readline().split())
    A[i-1] = x
    print(find_mex(A))