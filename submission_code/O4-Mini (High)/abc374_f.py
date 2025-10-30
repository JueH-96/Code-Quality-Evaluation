import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    X = int(next(it))
    # Read and store T[0..N-1]
    T = [int(next(it)) for _ in range(N)]
    # Prefix sums sumT[i] = sum of T[0..i-1]
    sumT = [0] * (N + 1)
    for i in range(N):
        sumT[i+1] = sumT[i] + T[i]
    INF = 10**30
    # dp[i]: dict mapping last shipment time L -> minimal total cost to ship first i orders
    dp = [dict() for _ in range(N+1)]
    # Initial state: before any shipment, set last shipment time L0 = -X so that L0+X = 0
    dp[0][-X] = 0

    for i in range(N+1):
        # Prune dp[i]: remove dominated states
        dict_i = dp[i]
        if not dict_i:
            # No way to reach this state
            continue
        # Sort by L ascending
        items = sorted(dict_i.items(), key=lambda kv: kv[0])
        # We won't need dp[i] dict anymore
        dp[i] = {}
        # Build the pruned list of states
        states = []
        bestCost = INF
        for L, c in items:
            # Keep only if cost strictly better than any seen so far
            if c < bestCost:
                states.append((L, c))
                bestCost = c

        # If we've shipped all N orders, we have our answer
        if i == N:
            # bestCost now is the minimal cost among final states
            print(bestCost)
            return

        # Otherwise, from each pruned state, try all next batches of size 1..K
        for L, c in states:
            Lpx = L + X
            maxb = N - i
            if maxb > K:
                maxb = K
            # Try batch sizes
            for b in range(1, maxb + 1):
                j = i + b
                # The release time of the last order in this batch
                Tj = T[j - 1]
                # Shipping time for this batch
                if Lpx >= Tj:
                    S = Lpx
                else:
                    S = Tj
                # Cost for this batch = sum(S - T[t]) for t in [i..j-1]
                costBatch = S * b - (sumT[j] - sumT[i])
                totalCost = c + costBatch
                # Update dp[j]
                dj = dp[j]
                prev = dj.get(S)
                if prev is None or totalCost < prev:
                    dj[S] = totalCost

    # If for some reason we exit loop without printing (shouldn't happen), print 0
    print(0)

# Call main to execute
main()