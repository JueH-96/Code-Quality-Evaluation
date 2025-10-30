import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1])
    A = list(map(int, data[2:]))

    # dp[i]: number of ways to split prefix of length i (1-based) with no segment summing to K,
    # where the last segment ends exactly at i.
    # We maintain prefix sums S[i] = sum A[1..i], with S[0] = 0.
    # Transition:
    #    dp[i] = sum_{j=0..i-1} dp[j]  minus sum_{j: S[j] = S[i]-K} dp[j]
    # Let acc[i] = sum_{j=0..i} dp[j].  Then total = acc[i-1].
    # We also keep a map cntMap[v] = sum of dp[j] for all j with S[j] = v.
    # Thus dp[i] = acc[i-1] - cntMap[S[i]-K].

    # Initialize
    dp0 = 1            # dp[0]
    acc = dp0          # acc[0] = dp[0]
    cntMap = {0: dp0}  # one way with prefix-sum 0 at position 0

    prefix_sum = 0
    dp_curr = 0

    for x in A:
        prefix_sum += x
        total = acc
        forbid = cntMap.get(prefix_sum - K, 0)
        dp_curr = (total - forbid) % MOD

        # update prefix sums and maps
        acc = (acc + dp_curr) % MOD
        cntMap[prefix_sum] = (cntMap.get(prefix_sum, 0) + dp_curr) % MOD

    # dp_curr is dp[N]
    print(dp_curr % MOD)

if __name__ == "__main__":
    main()