N, Q = map(int, input().split())
A = list(map(int, input().split()))

def find_mex(arr):
    seen = set(arr)
    mex = 0
    while mex in seen:
        mex += 1
    return mex

for _ in range(Q):
    i, x = map(int, input().split())
    A[i-1] = x
    print(find_mex(A))