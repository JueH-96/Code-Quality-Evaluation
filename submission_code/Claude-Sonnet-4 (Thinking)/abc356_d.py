N, M = map(int, input().split())
MOD = 998244353

result = 0

for bit_pos in range(60):  # Since N and M can be up to 2^60 - 1
    if M & (1 << bit_pos):  # If this bit is set in M
        cycle_length = 1 << (bit_pos + 1)  # 2^(bit_pos + 1)
        half_cycle = 1 << bit_pos          # 2^bit_pos
        
        full_cycles = (N + 1) // cycle_length
        remainder = (N + 1) % cycle_length
        
        count = (full_cycles % MOD) * (half_cycle % MOD) % MOD
        if remainder > half_cycle:
            count = (count + remainder - half_cycle) % MOD
        
        result = (result + count) % MOD

print(result)