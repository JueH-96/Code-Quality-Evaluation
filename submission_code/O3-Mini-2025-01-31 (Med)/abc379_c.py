def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
    except StopIteration:
        return
    
    # Read stone positions and counts.
    X = [int(next(it)) for _ in range(M)]
    A = [int(next(it)) for _ in range(M)]
    
    # Sort by cell index.
    paired = sorted(zip(X, A), key=lambda x: x[0])
    X = [p[0] for p in paired]
    A = [p[1] for p in paired]
    
    # Condition: total stones must equal N.
    if sum(A) != N:
        sys.stdout.write("-1")
        return
    
    # Also cell 1 must have at least one stone.
    if X[0] != 1:
        sys.stdout.write("-1")
        return

    # Compute prefix sums P[j] = A[0] + A[1] + ... + A[j].
    prefix = []
    s = 0
    for a in A:
        s += a
        prefix.append(s)
    
    # Helper function to compute sum of consecutive integers from L to R inclusive.
    def sum_range(L, R):
        n = R - L + 1
        return (L + R) * n // 2

    ans = 0
    # Process segments:
    # For j from 0 to M-1, the segment covers cells [X[j], R] where:
    #   R = X[j+1] - 1 (if j < M-1) and R = N - 1 (if j == M-1)
    for j in range(M):
        start = X[j]
        if j < M - 1:
            end = X[j+1] - 1
        else:
            end = N - 1  # Last segment covers cells from X[M-1] to N-1.
        
        # If start > end then there is no gap. We still need to check feasibility at cell 'start'.
        if start > end:
            if prefix[j] < start:
                sys.stdout.write("-1")
                return
            continue
        
        # Feasibility: at the worst-case index (largest i in the segment) we need P[j] >= end.
        if prefix[j] < end:
            sys.stdout.write("-1")
            return
        
        count = end - start + 1
        seg_sum = prefix[j] * count - sum_range(start, end)
        ans += seg_sum

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()