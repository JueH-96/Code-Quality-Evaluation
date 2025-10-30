# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

A = list(map(int, data[2:2+N]))
queries = list(zip(map(int, data[2+N::2]), map(int, data[3+N::2])))

mex_set = set(A)
mex = max(mex_set) + 1

for i, x in queries:
    mex_set.remove(A[i-1])
    mex_set.add(x)
    A[i-1] = x
    
    while mex in mex_set:
        mex += 1
    
    print(mex)