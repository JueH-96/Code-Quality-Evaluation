def count_strings(K, C):
    MOD = 998244353
    total_count = 0
    
    # Calculate the total number of strings of length 1 to K
    for length in range(1, K + 1):
        # Calculate the number of valid strings of this length
        count_for_length = 1
        for c in C:
            count_for_length *= (c + 1)  # Each letter can appear from 0 to C_i times
            count_for_length %= MOD
        
        total_count += count_for_length
        total_count %= MOD
    
    return total_count

# Read input
import sys
input = sys.stdin.read
data = input().split()

K = int(data[0])
C = list(map(int, data[1:27]))

# Get the result
result = count_strings(K, C)

# Print the result
print(result)