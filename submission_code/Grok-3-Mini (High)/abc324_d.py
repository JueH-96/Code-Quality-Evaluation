import sys
import math

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]

# Create a sorted tuple of the digits in S
sorted_S = tuple(sorted(S))

# Calculate the maximum number with N digits
max_num = 10**N - 1

# Calculate the maximum K such that K*K <= max_num
max_K = int(math.sqrt(max_num))

# Initialize count of square numbers
count = 0

# Iterate over all possible K from 0 to max_K
for K in range(0, max_K + 1):
    # Compute the square
    num = K * K
    # Convert to string and pad with leading zeros to length N
    num_str = str(num).zfill(N)
    # Sort the digits and check if they match the sorted digits of S
    if tuple(sorted(num_str)) == sorted_S:
        count += 1

# Output the count
print(count)