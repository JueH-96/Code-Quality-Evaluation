import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = [0] * (N + 1)
    B = [0] * (N)      # B[1..N-1]
    for i in range(1, N+1):
        A[i] = int(next(it))
    for i in range(1, N):
        B[i] = int(next(it))
    # sort
    A_sorted = [0] + sorted(A[1:])
    B_sorted = [0] + sorted(B[1:])  # B_sorted[1..N-1]
    # prefix_valid[i] = True if for all j<=i: A[j] <= B[j]
    prefix_valid = [True] * (N+1)
    for i in range(1, N):
        if prefix_valid[i-1] and A_sorted[i] <= B_sorted[i]:
            prefix_valid[i] = True
        else:
            prefix_valid[i] = False
    # suffix_valid[pos] = True if for all i in pos+1..N: A[i] <= B[i-1]
    suffix_valid = [True] * (N+2)
    # suffix_valid[N] = True
    for pos in range(N-1, 0, -1):
        # check A[pos+1] <= B[pos]
        if suffix_valid[pos+1] and A_sorted[pos+1] <= B_sorted[pos]:
            suffix_valid[pos] = True
        else:
            suffix_valid[pos] = False
    INF = 10**30
    ans = INF
    # Try inserting x as position pos in 1..N
    for pos in range(1, N+1):
        if not prefix_valid[pos-1]:
            continue
        if not suffix_valid[pos]:
            continue
        # compute L and R for x
        L = A_sorted[pos]
        if pos >= 2:
            # x must be > B_sorted[pos-1]
            L = max(L, B_sorted[pos-1] + 1)
        if pos <= N-1:
            R = B_sorted[pos]
        else:
            R = INF
        if L <= R:
            ans = min(ans, L)
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()