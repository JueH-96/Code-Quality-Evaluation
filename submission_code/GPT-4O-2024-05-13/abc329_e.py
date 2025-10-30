# YOUR CODE HERE
def can_match(N, M, S, T):
    # Create a list to keep track of which positions in S can be matched by T
    matchable = [False] * N
    
    # Check each position in S to see if T can be placed there
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            for j in range(M):
                matchable[i+j] = True
    
    # Check if all characters in S can be matched
    for i in range(N):
        if not matchable[i]:
            return "No"
    
    return "Yes"

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
S = data[2]
T = data[3]

print(can_match(N, M, S, T))