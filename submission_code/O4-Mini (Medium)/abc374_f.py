def main():
    import sys
    input = sys.stdin.readline
    N, K, X = map(int, input().split())
    T = list(map(int, input().split()))
    # T is sorted by problem statement
    # prefix sums of T for cost computation
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + T[i]
    # dp[i]: dict mapping last shipment day S_prev to minimal cost
    # after shipping first i orders
    INF = 10**40
    dp = [dict() for _ in range(N+1)]
    # initialize before any shipping: set S_prev = -X so that first batch ships at T[j]
    dp[0][-X] = 0
    for i in range(N):
        if not dp[i]:
            continue
        # snapshot items to avoid mutation during iteration
        items = list(dp[i].items())
        for S_prev, cost_prev in items:
            # try batch sizes from 1 to K
            # batch covers orders i..j-1 in 0-based, i+1..j in 1-based
            # so j runs from i+1 to min(N, i+K)
            max_j = i + K
            if max_j > N:
                max_j = N
            # we will accumulate sumT on the fly via prefix sums
            for j in range(i+1, max_j+1):
                size = j - i
                Tj = T[j-1]
                # earliest due to cool-down
                earliest = S_prev + X
                # actual shipment day
                if earliest > Tj:
                    S_batch = earliest
                else:
                    S_batch = Tj
                # sum of T for orders i+1..j is prefix[j] - prefix[i]
                sumT = prefix[j] - prefix[i]
                # cost for this batch
                cost_batch = size * S_batch - sumT
                new_cost = cost_prev + cost_batch
                # update dp[j] at S_batch
                prev = dp[j].get(S_batch)
                if prev is None or new_cost < prev:
                    dp[j][S_batch] = new_cost
    # answer is minimal cost over all end states at dp[N]
    ans = min(dp[N].values())
    print(ans)

if __name__ == "__main__":
    main()