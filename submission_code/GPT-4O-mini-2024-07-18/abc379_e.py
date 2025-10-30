def sum_of_substrings(N, S):
    total_sum = 0
    current_sum = 0
    multiplier = 1
    
    # Iterate from the end of the string to the beginning
    for i in range(N - 1, -1, -1):
        digit = int(S[i])
        current_sum = current_sum + digit * multiplier
        total_sum += current_sum
        multiplier = multiplier * 10 + 1  # Update multiplier for the next digit
    
    return total_sum

import sys

# Read input
input_data = sys.stdin.read().strip().split()
N = int(input_data[0])
S = input_data[1]

# Calculate the result
result = sum_of_substrings(N, S)

# Print the result
print(result)