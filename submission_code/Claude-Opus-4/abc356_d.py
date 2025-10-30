def count_ones_at_bit(n, bit):
    """Count how many numbers from 0 to n have bit 'bit' set to 1"""
    cycle_length = 1 << (bit + 1)  # 2^(bit+1)
    complete_cycles = (n + 1) // cycle_length
    remainder = (n + 1) % cycle_length
    
    count = complete_cycles * (1 << bit)  # complete_cycles * 2^bit
    
    if remainder > (1 << bit):
        count += remainder - (1 << bit)
    
    return count

def solve(N, M):
    MOD = 998244353
    result = 0
    
    # Check each bit position
    for bit in range(60):  # N and M are at most 2^60 - 1
        if M & (1 << bit):  # If bit 'bit' is set in M
            count = count_ones_at_bit(N, bit)
            result = (result + count) % MOD
    
    return result

# Read input
N, M = map(int, input().split())

# Solve and print result
print(solve(N, M))