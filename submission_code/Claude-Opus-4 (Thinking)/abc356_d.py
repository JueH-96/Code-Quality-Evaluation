# YOUR CODE HERE
n, m = map(int, input().split())
MOD = 998244353

total = 0
for bit_pos in range(60):
    if m & (1 << bit_pos):
        # Count how many numbers from 0 to n have bit bit_pos set to 1
        cycle_len = 1 << (bit_pos + 1)
        half_cycle = 1 << bit_pos
        
        full_cycles = (n + 1) // cycle_len
        remaining = (n + 1) % cycle_len
        
        count = full_cycles * half_cycle
        if remaining > half_cycle:
            count += remaining - half_cycle
        
        total = (total + count) % MOD

print(total)