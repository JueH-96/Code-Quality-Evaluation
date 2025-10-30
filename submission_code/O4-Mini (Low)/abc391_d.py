import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N, W = map(int, input().split())
    cols = [[] for _ in range(W+1)]
    Y = [0]*(N+1)
    X = [0]*(N+1)
    for i in range(1, N+1):
        x,y = map(int, input().split())
        X[i] = x
        Y[i] = y
        cols[x].append((y, i))
    # sort each column by ascending Y
    C = float('inf')
    for x in range(1, W+1):
        cols[x].sort()
        C = min(C, len(cols[x]))
    if C == float('inf'):  # no columns?
        C = 0
    # compute for k=1..C the removal times
    # T_k = max over columns x of (Y_xk - k)
    # we'll build A[1..C]
    A = [-10**30] * (C+1)
    for x in range(1, W+1):
        # for k in this column, update
        cnt = len(cols[x])
        # only up to C
        lim = min(cnt, C)
        for k in range(1, lim+1):
            y, idx = cols[x][k-1]
            val = y - k
            if val > A[k]:
                A[k] = val
    # Now assign removal_time for each block
    INF = 10**30
    removal = [INF] * (N+1)
    # for each column, the k-th block gets removal time T_k = A[k]
    for x in range(1, W+1):
        cnt = len(cols[x])
        for k in range(1, cnt+1):
            y, idx = cols[x][k-1]
            if k <= C:
                removal[idx] = A[k]
            else:
                removal[idx] = INF
    # process queries
    Q = int(input())
    out = []
    for _ in range(Q):
        t, a = map(int, input().split())
        # exists at time t+0.5 iff t < removal_time
        if t < removal[a]:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()