import sys
data = sys.stdin.read().split()
N = int(data[0])
M = int(data[1])
MOD = 998244353
sum_ans = 0
for i in range(60):
    if (M & (1 << i)) != 0:
        P = 1 << (i + 1)
        bit_value = 1 << i
        num_complete = N // P
        count_complete_mod = ((num_complete % MOD) * (bit_value % MOD)) % MOD
        R = N % P
        if R >= bit_value:
            count_remainder = R - bit_value + 1
        else:
            count_remainder = 0
        count_remainder_mod = count_remainder % MOD
        total_count_mod = (count_complete_mod + count_remainder_mod) % MOD
        sum_ans = (sum_ans + total_count_mod) % MOD
print(sum_ans)