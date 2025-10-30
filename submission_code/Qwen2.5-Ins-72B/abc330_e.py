# YOUR CODE HERE
import sys
from collections import defaultdict

def mex(arr):
    seen = set(arr)
    mex = 0
    while mex in seen:
        mex += 1
    return mex

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    A = list(map(int, data[2:2+N]))
    queries = [(int(data[2+N+i*2])-1, int(data[3+N+i*2])) for i in range(Q)]
    
    for i, x in queries:
        A[i] = x
        print(mex(A))

if __name__ == "__main__":
    main()