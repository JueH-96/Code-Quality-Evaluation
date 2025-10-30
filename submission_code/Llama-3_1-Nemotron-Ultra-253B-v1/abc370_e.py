MOD = 998244353

n, k = map(int, input().split())
a = list(map(int, input().split()))

prefix = [0] * (n + 1)
for i in range(1, n + 1):
    prefix[i] = prefix[i-1] + a[i-1]

from collections import defaultdict
prefix_map = defaultdict(int)
prefix_map[0] = 1  # S[0] = 0, dp[0] = 1
sum_so_far = 1

current_dp = 0
for i in range(1, n + 1):
    current_S = prefix[i]
    target = current_S - k
    current_dp = (sum_so_far - prefix_map.get(target, 0)) % MOD
    sum_so_far = (sum_so_far + current_dp) % MOD
    prefix_map[current_S] = (prefix_map.get(current_S, 0) + current_dp) % MOD

print(current_dp % MOD)