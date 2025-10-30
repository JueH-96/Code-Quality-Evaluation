mod = 998244353

N, M = map(int, input().split())

sum_total = 0

for i in range(60):
    if (M >> i) & 1:
        pow_i_plus_1 = (1 << (i + 1))
        pow_i = (1 << i)
        full_cycles = (N + 1) // pow_i_plus_1
        count = full_cycles * pow_i
        remainder = (N + 1) % pow_i_plus_1
        if remainder > pow_i:
            count += remainder - pow_i
        sum_total += count

sum_total %= mod
print(sum_total)