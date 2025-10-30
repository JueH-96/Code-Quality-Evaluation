def count_set_bits(N, bit_pos):
    """Count how many numbers in [0, N] have bit_pos-th bit set to 1"""
    if N < 0:
        return 0
    
    period = 1 << (bit_pos + 1)  # 2^(bit_pos + 1)
    half_period = 1 << bit_pos   # 2^bit_pos
    
    complete_periods = (N + 1) // period
    remainder = (N + 1) % period
    
    # Each complete period contributes half_period set bits
    result = complete_periods * half_period
    
    # Handle the partial period
    if remainder > half_period:
        result += remainder - half_period
    
    return result

N, M = map(int, input().split())

MOD = 998244353
total = 0

# Check each bit position
for i in range(60):  # 60 bits is enough for 2^60 - 1
    if M & (1 << i):  # If i-th bit of M is 1
        count = count_set_bits(N, i)
        total = (total + count) % MOD

print(total)