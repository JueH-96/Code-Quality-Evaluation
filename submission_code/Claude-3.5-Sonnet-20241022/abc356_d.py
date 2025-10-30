def popcount(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

N, M = map(int, input().split())
MOD = 998244353

# For each bit position in M that is 1, we need to count how many numbers
# from 0 to N will have that bit set after AND with M
result = 0

# For each number k from 0 to N, k & M will give a number whose bits are subset of M's bits
# So we only need to consider bit positions where M has 1
curr_M = M
while curr_M:
    # Get least significant bit of M
    bit = curr_M & -curr_M
    
    # Count how many numbers from 0 to N will have this bit set after AND with M
    # This is equivalent to counting numbers that have 1 in this position and
    # are less than or equal to N
    cycle = bit * 2  # Length of one complete cycle (0s and 1s)
    complete = N // cycle  # Number of complete cycles
    remainder = N % cycle  # Remaining numbers
    
    # Count numbers with this bit set
    count = complete * bit  # From complete cycles
    if remainder >= bit:  # From remainder
        count += remainder - bit + 1
        
    # Add this bit's contribution to result
    result = (result + count) % MOD
    
    # Remove the processed bit
    curr_M &= curr_M - 1

print(result)