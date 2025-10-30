from collections import Counter

N, Q = map(int, input().split())
A = list(map(int, input().split()))

for _ in range(Q):
    i, x = map(int, input().split())
    A[i-1] = x
    
    mex = 0
    while mex in A:
        mex += 1
    
    print(mex)