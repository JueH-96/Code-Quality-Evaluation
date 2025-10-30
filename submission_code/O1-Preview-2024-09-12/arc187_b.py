# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    mod = 998244353

    NINF = -float('inf')

    # dp[i][min_v][num_unique] = number of ways to fill first i positions with minimal value min_v, and num_unique unique values >= min_v
    dp = [{} for _ in range(N + 1)]
    dp[0][(M+1, 0)] = 1  # Start with min_v=M+1 (impossible), num_unique=0

    for i in range(N):
        next_dp = {}
        S_i = [B[i]] if B[i] != -1 else range(1, M + 1)
        for (min_v, num_unique), cnt in dp[i].items():
            for v in S_i:
                new_min_v = min(min_v, v)
                new_num_unique = num_unique
                if v >= new_min_v:
                    new_num_unique = num_unique + (1 if v != min_v else 0)
                key = (new_min_v, new_num_unique)
                next_dp[key] = (next_dp.get(key, 0) + cnt) % mod
        dp[i + 1] = next_dp

    ans = 0
    for (min_v, num_unique), cnt in dp[N].items():
        if min_v <= M:
            # The number of connected components is equal to the number of unique values ≥ min_v
            num_components = num_unique
            ans = (ans + cnt * num_components) % mod
    print(ans)

threading.Thread(target=main).start()