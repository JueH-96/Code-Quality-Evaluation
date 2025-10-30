def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    D = int(next(it))
    
    # Read gift values for Aoki and Snuke
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # Sort the candidates for Snuke for binary searching
    B.sort()
    
    best = -1  # Initialize answer as -1 (no valid pair found yet)
    
    # For each gift candidate a for Aoki,
    # the gift for Snuke must be in the range [a - D, a + D] to satisfy |a - b| <= D.
    # To maximize the sum a+b, we choose the maximum possible b from B that is <= a+D.
    for a in A:
        # Find the insertion index in B for a+D such that all elements before are <= a+D.
        idx = bisect.bisect_right(B, a + D)
        if idx == 0:
            continue
        candidate = B[idx - 1]
        # Check if candidate meets the lower bound condition: candidate >= a-D.
        if candidate < a - D:
            continue
        # Update best sum if this pair gives a higher sum.
        total = a + candidate
        if total > best:
            best = total
            
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()