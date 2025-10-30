def main():
    import sys
    from bisect import bisect_left
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    
    # Read the (a_i, b_i) pairs
    pairs = []
    idx = 2
    for _ in range(N):
        a_i = int(data[idx])
        b_i = int(data[idx+1])
        idx += 2
        pairs.append((a_i, b_i))
    
    # Sort by a_i
    pairs.sort(key=lambda x: x[0])
    
    # Build a list of a_i and prefix sums of b_i
    sorted_a = [p[0] for p in pairs]
    prefix_sum = [0] * (N+1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + pairs[i-1][1]
    
    # Function to get total pills on day d
    # sum of b_i for all i with a_i >= d
    def pills(d):
        i = bisect_left(sorted_a, d)
        return prefix_sum[-1] - prefix_sum[i]
    
    # We know the first day with sum <= K is in [1, maxA+1]
    maxA = sorted_a[-1]
    lo, hi = 1, maxA + 1
    
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        if pills(mid) <= K:
            hi = mid
        else:
            lo = mid + 1
    
    print(lo)

# Do not forget to call main() at the end
main()