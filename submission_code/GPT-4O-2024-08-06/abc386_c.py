# YOUR CODE HERE
def can_transform(S, T):
    if S == T:
        return "Yes"
    
    len_s = len(S)
    len_t = len(T)
    
    # Check for replacement
    if len_s == len_t:
        # Count the number of differing characters
        diff_count = sum(1 for a, b in zip(S, T) if a != b)
        if diff_count == 1:
            return "Yes"
    
    # Check for insertion
    if len_s + 1 == len_t:
        # Try to find the position where the extra character is in T
        for i in range(len_t):
            if S[:i] == T[:i] and S[i:] == T[i+1:]:
                return "Yes"
    
    # Check for deletion
    if len_s == len_t + 1:
        # Try to find the position where the extra character is in S
        for i in range(len_s):
            if S[:i] == T[:i] and S[i+1:] == T[i:]:
                return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().strip().split()
K = int(data[0])
S = data[1]
T = data[2]

print(can_transform(S, T))