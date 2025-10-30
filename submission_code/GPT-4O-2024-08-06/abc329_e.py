# YOUR CODE HERE
def can_transform(N, M, S, T):
    # Create a boolean array to mark which positions can be filled by T
    can_fill = [False] * N
    
    # Check each possible starting position for T in S
    for start in range(N - M + 1):
        # Check if T matches the substring of S starting at `start`
        if S[start:start + M] == T:
            # Mark these positions as fillable
            for i in range(M):
                can_fill[start + i] = True
    
    # Check if all positions in S can be filled
    for i in range(N):
        if not can_fill[i]:
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]
T = data[3]

# Output the result
print(can_transform(N, M, S, T))