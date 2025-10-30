def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0]); K = int(data[1]); X = int(data[2])
    T = list(map(int, data[3:3+N]))
    
    # Precompute prefix sums so that we can quickly compute the sum of T[i:j]
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + T[i]
    
    # We use memoized recursion (dp) where f(i, L) gives the minimum additional dissatisfaction 
    # for scheduling orders[i]...orders[N-1] when the next shipment can be made on at least day L.
    #
    # At each step, we can group up to K orders in one shipment. For a given group, suppose we take
    # orders i, i+1, ..., i+m-1 with 1 <= m <= K. The shipment day must be at least L (due to the waiting X days
    # rule from the last shipment) and also at least T[i+m-1] (because each order can only be shipped on or after its T).
    # Therefore, the optimal shipment day to minimize the waiting is d = max(L, T[i+m-1]).
    #
    # The dissatisfaction incurred for these orders is:
    #    sum_{k=i}^{i+m-1} (d - T[k]) = m*d - (prefix[i+m] - prefix[i])
    #
    # After shipping this group on day d, the next shipment can be made on day d + X.
    
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def f(i, L):
        if i == N:
            return 0
        best = float('inf')
        # Try grouping m orders, where m can be from 1 to K (as long as we have enough orders).
        for m in range(1, K+1):
            if i + m > N:
                break
            # The shipment day we choose must be at least L and T[i+m-1] (the latest order's placement day).
            d = L if L > T[i+m-1] else T[i+m-1]
            # Compute dissatisfaction for orders i through i+m-1.
            cost = m * d - (prefix[i+m] - prefix[i])
            best = min(best, cost + f(i+m, d + X))
        return best
    
    ans = f(0, 1)
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()