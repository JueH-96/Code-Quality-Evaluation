def is_11_22_string(N, S):
    # Check if the length is odd
    if N % 2 == 0:
        return "No"
    
    mid_index = N // 2  # This is the index of the '/' character
    
    # Check the first half for '1's
    for i in range(mid_index):
        if S[i] != '1':
            return "No"
    
    # Check the middle character for '/'
    if S[mid_index] != '/':
        return "No"
    
    # Check the second half for '2's
    for i in range(mid_index + 1, N):
        if S[i] != '2':
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
S = data[1]

# Output the result
print(is_11_22_string(N, S))