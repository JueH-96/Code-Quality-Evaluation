import collections
import sys

# Global precomputation
MAX_A_VAL = 200000 # Max value of A_i

# Precompute Smallest Prime Factor (spf)
# spf_global[i] will store the smallest prime factor of i
spf_global = list(range(MAX_A_VAL + 1)) # Initialize spf_global[i] = i

# Iterate from 2 up to sqrt(MAX_A_VAL)
# For each prime i, mark its multiples
# This loop correctly computes SPF for all numbers up to MAX_A_VAL.
# For a number j, its SPF spf_global[j] is set to i if i is the smallest prime dividing j.
# If spf_global[j] == j, it means j has not been marked by a smaller prime factor yet.
# Thus, if i is prime (spf_global[i] == i), and j is a multiple of i (j = i*k),
# then if spf_global[j] is still j, it implies i is the smallest prime factor of j.
for i in range(2, int(MAX_A_VAL**0.5) + 1):
    if spf_global[i] == i: # i is prime
        for j in range(i * i, MAX_A_VAL + 1, i):
            if spf_global[j] == j: # If not already marked by a smaller prime
                spf_global[j] = i

# Precompute square-free part for each number up to MAX_A_VAL
sf_part_global = [0] * (MAX_A_VAL + 1)
if MAX_A_VAL >= 1:
    sf_part_global[1] = 1 # Base case: square-free part of 1 is 1

for i in range(2, MAX_A_VAL + 1):
    p = spf_global[i]       # Smallest prime factor of i
    remainder_val = i // p  # i = p * remainder_val
    
    sf_of_remainder = sf_part_global[remainder_val]
    
    if sf_of_remainder % p == 0:
        sf_part_global[i] = sf_of_remainder // p
    else:
        sf_part_global[i] = sf_of_remainder * p

def solve():
    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().split()
    A = [int(x) for x in A_str] # Using list comprehension for potentially faster parsing

    num_zeros = 0
    sf_counts = collections.defaultdict(int)

    for x_val in A:
        if x_val == 0:
            num_zeros += 1
        else:
            sf_counts[sf_part_global[x_val]] += 1
            
    ans = 0

    # Case 1: Pairs involving at least one zero
    # Number of pairs (0,0): num_zeros * (num_zeros - 1) // 2
    # Number of pairs (0, non-zero): num_zeros * (N - num_zeros)
    ans += num_zeros * (num_zeros - 1) // 2
    ans += num_zeros * (N - num_zeros)
    
    # Case 2: Pairs of two non-zero numbers
    for s_val_key in sf_counts: # Iterate through distinct square-free parts found
        count = sf_counts[s_val_key]
        ans += count * (count - 1) // 2
        
    print(ans)

if __name__ == '__main__':
    solve()