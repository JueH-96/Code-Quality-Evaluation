# Read the input
N = int(input())

# Initialize the sum
total = 0

# Generate all permutations of (1, 2, ..., N)
from itertools import permutations
for p in permutations(range(1, N+1)):
    # Compute f(P)
    f_p = int(''.join(str(x) for x in p))
    # Add f(P) to the total
    total += f_p

# Print the result modulo 998244353
print(total % 998244353)