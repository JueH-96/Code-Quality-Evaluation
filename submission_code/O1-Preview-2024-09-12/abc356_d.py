# YOUR CODE HERE
N_str, M_str = input().split()
N = int(N_str)
M = int(M_str)
mod = 998244353

total_sum = 0
N_plus_one = N + 1
for b in range(61):  # since N and M can be up to 2^60 -1
    if (M >> b) & 1:
        cycle_length = 1 << (b + 1)
        full_cycles = N_plus_one // cycle_length
        remainder = N_plus_one % cycle_length
        ones_in_full_cycles = full_cycles * (1 << b)
        ones_in_remainder = max(0, remainder - (1 << b))
        count = ones_in_full_cycles + ones_in_remainder
        total_sum = (total_sum + count) % mod

print(total_sum)