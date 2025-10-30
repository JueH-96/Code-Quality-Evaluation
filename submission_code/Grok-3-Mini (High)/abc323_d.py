import sys
from collections import defaultdict

# Read all input and convert to integers
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1

# Dictionary to hold sum of contributions for each odd part
sum_dict = defaultdict(int)

# Process each slime size and count
for _ in range(N):
    S_val = data[index]  # Size S_i
    C_val = data[index + 1]  # Count C_i
    index += 2
    
    # Compute the exponent E and odd part O
    temp = S_val
    E = 0
    while temp % 2 == 0:
        temp //= 2
        E += 1
    O = temp  # O is the odd part
    
    # Compute contribution: C_val * (2 ** E)
    contrib = C_val * (1 << E)
    
    # Add to the sum for the odd part O
    sum_dict[O] += contrib

# Compute the total minimum number of slimes
total_min = 0
for s_sum in sum_dict.values():
    # Population count (number of 1s in binary representation)
    pop_count = bin(s_sum).count('1')
    total_min += pop_count

# Output the result
print(total_min)