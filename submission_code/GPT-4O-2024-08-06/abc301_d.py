# YOUR CODE HERE
def find_max_binary(S, N):
    # Convert S to a list for mutability
    S = list(S)
    
    # Iterate over each character in S
    for i in range(len(S)):
        if S[i] == '?':
            # Try to replace '?' with '1' first
            S[i] = '1'
            # Convert the current binary string to a decimal number
            current_value = int(''.join(S), 2)
            # If this value exceeds N, revert this position to '0'
            if current_value > N:
                S[i] = '0'
    
    # Convert the final binary string to a decimal number
    final_value = int(''.join(S), 2)
    
    # Check if the final value is within the limit
    if final_value <= N:
        return final_value
    else:
        return -1

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
N = int(data[1])

# Get the result
result = find_max_binary(S, N)

# Print the result
print(result)