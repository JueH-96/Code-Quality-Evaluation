MOD = 998244353

N, K = map(int, input().split())
A = list(map(int, input().split()))

prefix_sum = [0] * (N + 1)
for i in range(1, N + 1):
    prefix_sum[i] = prefix_sum[i-1] + A[i-1]

from collections import defaultdict
sum_map = defaultdict(int)
sum_map[0] = 1
sum_dp = 1
dp_i = 0

for i in range(1, N + 1):
    current_prefix = prefix_sum[i]
    target = current_prefix - K
    forbidden = sum_map.get(target, 0)
    dp_i = (sum_dp - forbidden) % MOD
    sum_map[current_prefix] = (sum_map[current_prefix] + dp_i) % MOD
    sum_dp = (sum_dp + dp_i) % MOD

print(dp_i % MOD)