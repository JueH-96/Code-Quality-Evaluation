MOD = 998244353

N, M = map(int, input().split())

sum_total = 0
for i in range(60):
    if (M >> i) & 1:
        two_i = 1 << i
        two_i_plus_1 = two_i << 1
        full_cycles = (N + 1) // two_i_plus_1
        count = full_cycles * two_i
        remainder = (N + 1) % two_i_plus_1
        if remainder > two_i:
            count += remainder - two_i
        sum_total += count
        sum_total %= MOD

print(sum_total % MOD)