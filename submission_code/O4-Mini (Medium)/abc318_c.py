def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    D = int(next(it))
    P = int(next(it))
    F = [int(next(it)) for _ in range(N)]
    
    # Total regular fare if we buy no passes
    total_fare = sum(F)
    
    # Sort fares descending so we can cover the most expensive days first
    F.sort(reverse=True)
    
    # Build prefix sums of sorted fares: prefix[i] = sum of top i fares
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + F[i]
    
    # We'll try buying b batches, for b from 0 to ceil(N/D)
    # Each batch gives us D passes, so b batches = b*D passes
    # We can cover at most min(b*D, N) days, saving prefix[min(b*D,N)] in fares
    # Cost = b*P + (total_fare - saved)
    max_batches = (N + D - 1) // D
    ans = total_fare  # case b = 0
    
    for b in range(1, max_batches + 1):
        passes = b * D
        used = passes if passes <= N else N
        saved = prefix[used]
        cost = b * P + (total_fare - saved)
        if cost < ans:
            ans = cost
    
    print(ans)

if __name__ == "__main__":
    main()