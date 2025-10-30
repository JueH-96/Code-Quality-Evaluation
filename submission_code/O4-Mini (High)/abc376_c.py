import sys
from bisect import bisect_left

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N-1)]
    
    # Sort toy sizes and existing box sizes
    A.sort()
    B.sort()
    
    # okL[i] = True iff for all j< i, A[j] <= B[j]
    okL = [False] * N
    okL[0] = True
    for i in range(1, N):
        okL[i] = okL[i-1] and (A[i-1] <= B[i-1])
    
    # okR[i] = True iff for all j> i, A[j] <= B[j-1]
    okR = [False] * N
    okR[N-1] = True
    for i in range(N-2, -1, -1):
        okR[i] = okR[i+1] and (A[i+1] <= B[i])
    
    # f(x): can we match all toys if purchased box has size x?
    # Compute insertion position ins = bisect_left(B, x).
    # Then we need
    #   okL[ins] == True,
    #   okR[ins] == True,
    #   A[ins] <= x.
    INF = 10**9 + 1
    lo, hi = 1, INF
    while lo < hi:
        mid = (lo + hi) // 2
        ins = bisect_left(B, mid)
        if okL[ins] and okR[ins] and A[ins] <= mid:
            hi = mid
        else:
            lo = mid + 1
    
    # If lo reached INF, no feasible x in [1..1e9]
    if lo == INF:
        print(-1)
    else:
        print(lo)

if __name__ == "__main__":
    main()