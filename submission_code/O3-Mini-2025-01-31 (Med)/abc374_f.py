def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    X = int(next(it))
    T = [int(next(it)) for _ in range(N)]
    
    # dp[i] will be a dictionary mapping "last shipment day" used so far to
    # the minimal total dissatisfaction when processing the first i orders.
    # For the first shipment, we have no previous shipment day restriction.
    # To simulate this uniformly, we set the "previous shipment day" as -X.
    # Then for the first shipment S, 
    # we require S >= -X+X = 0 and also S >= T[0] so that S will simply be set to T[0] if T[0] > 0.
    dp = [dict() for _ in range(N + 1)]
    dp[0][-X] = 0  # initial state: no orders processed, and pretend last shipment day was -X.
    
    # To quickly compute sums over intervals, compute prefix sums for T.
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + T[i]
    
    # Process orders in a dynamic programming manner.
    # For each state (i orders processed, with "last shipment day" = last),
    # consider grouping from 1 to K orders into the next shipment.
    # Orders are considered in their given (sorted) order.
    for i in range(N):
        for last, cost in dp[i].items():
            # Try grouping m orders (m from 1 to K) starting at index i.
            for m in range(1, K + 1):
                if i + m > N:
                    break
                # Group covers orders from index i to i+m-1.
                # All orders in the group must be shipped on day S.
                # S must be at least the maximum of:
                #  - last shipment day + X (if there was a previous shipment)
                #  - T[i+m-1] (the latest order placement date in this group)
                # Choosing S to be exactly the minimum allowed minimizes additional dissatisfaction.
                S = max(last + X, T[i + m - 1])
                # The cost incurred for the orders in this group equals:
                #   m * S - (T[i] + T[i+1] + ... + T[i+m-1])
                group_cost = m * S - (prefix[i + m] - prefix[i])
                new_cost = cost + group_cost
                # Update the state for having processed i+m orders, with last shipment day S.
                if S in dp[i + m]:
                    if new_cost < dp[i + m][S]:
                        dp[i + m][S] = new_cost
                else:
                    dp[i + m][S] = new_cost

    # The final answer is the minimum dissatisfaction among all states that have processed all N orders.
    ans = min(dp[N].values()) if dp[N] else 0
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()