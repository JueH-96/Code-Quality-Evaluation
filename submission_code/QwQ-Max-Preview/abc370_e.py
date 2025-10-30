n, K = map(int, input().split())
A = list(map(int, input().split()))

s = [0] * (n + 1)
for i in range(1, n + 1):
    s[i] = s[i-1] + A[i-1]

MOD = 998244353

dp_prev = 1
prefix_sum_prev = 1
current_cnt = {s[0]: dp_prev}

for i in range(1, n + 1):
    current_s = s[i]
    target = current_s - K
    sum_dp_j = current_cnt.get(target, 0)
    dp_i = (prefix_sum_prev - sum_dp_j) % MOD
    new_prefix_sum = (prefix_sum_prev + dp_i) % MOD
    
    if current_s in current_cnt:
        current_cnt[current_s] = (current_cnt[current_s] + dp_i) % MOD
    else:
        current_cnt[current_s] = dp_i
    
    dp_prev, prefix_sum_prev = dp_i, new_prefix_sum

print(dp_prev % MOD)