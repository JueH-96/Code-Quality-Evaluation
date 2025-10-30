import sys
import threading

def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        print(-1)
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    X = [0]*M
    for i in range(M):
        X[i] = int(next(it))
    A = [0]*M
    for i in range(M):
        A[i] = int(next(it))

    # Total stones must equal N
    total = 0
    for a in A:
        total += a
    if total != N:
        print(-1)
        return

    # Pair and sort by position
    pairs = list(zip(X, A))
    pairs.sort(key=lambda x: x[0])
    # After sort, check first position is 1
    if pairs[0][0] != 1:
        print(-1)
        return

    # Check prefix feasibility:
    # Let P[i] = sum of A up to i-th sorted source.
    # For each gap before next source at pos X_next,
    # we need P[i] >= X_next - 1.
    pref = 0
    for i in range(M - 1):
        xi, ai = pairs[i]
        pref += ai
        xnext, _ = pairs[i+1]
        if pref < xnext - 1:
            print(-1)
            return
    # Also include last group to complete pref (not really needed for checks)
    # but let's do it to compute cost below.
    # Actually cost uses total and sum_xa only.
    # Compute cost = sum_{j=1..N} j - sum_i Xi*Ai
    # sum_j = N*(N+1)//2
    sum_j = N*(N+1) // 2
    sum_xa = 0
    for xi, ai in pairs:
        sum_xa += xi * ai

    ans = sum_j - sum_xa
    print(ans)

if __name__ == "__main__":
    main()