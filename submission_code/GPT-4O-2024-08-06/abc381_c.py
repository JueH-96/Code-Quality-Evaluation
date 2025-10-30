# YOUR CODE HERE
def find_max_11_22_string_length(N, S):
    max_length = 0
    
    for i in range(N):
        if S[i] == '/':
            # Try to expand around this '/' character
            left = i - 1
            right = i + 1
            current_length = 1  # Start with the '/' itself
            
            # Expand symmetrically
            while left >= 0 and right < N and S[left] == '1' and S[right] == '2':
                current_length += 2
                left -= 1
                right += 1
            
            # Check if the length is odd (it should be by construction)
            if current_length % 2 == 1:
                max_length = max(max_length, current_length)
    
    return max_length

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Find and print the result
result = find_max_11_22_string_length(N, S)
print(result)