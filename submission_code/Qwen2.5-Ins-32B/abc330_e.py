import sys

def find_mex(arr):
    mex = 0
    while mex in arr:
        mex += 1
    return mex

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n, q = int(data[0]), int(data[1])
    A = list(map(int, data[2:n+2]))
    queries = list(zip(map(int, data[n+2::2]), map(int, data[n+3::2])))
    
    for i, x in queries:
        A[i-1] = x
        print(find_mex(A))

solve()