import sys

def is_airport_code(S, T):
    # Convert T to lowercase for comparison
    T = T.lower()
    
    # Check if T is a subsequence of S
    def is_subsequence(s, t):
        t_iter = iter(t)
        return all(char in t_iter for char in s)
    
    # Check if T can be formed by a subsequence of length 3
    if is_subsequence(T, S):
        return "Yes"
    
    # Check if T can be formed by a subsequence of length 2 + 'X'
    if T[2] == 'x' and is_subsequence(T[:2], S):
        return "Yes"
    
    return "No"

# Read input from stdin
S = input().strip()
T = input().strip()

# Determine if T is an airport code for S
result = is_airport_code(S, T)

# Write the answer to stdout
print(result)