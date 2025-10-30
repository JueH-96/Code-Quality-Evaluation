def can_form_string(N, M, S, T):
    # Create a list to track the positions in S where T can match
    match_positions = [0] * (N - M + 1)
    
    # Check all possible starting positions in S for T
    for i in range(N - M + 1):
        if S[i:i + M] == T:
            match_positions[i] = 1
    
    # Now we need to check if we can cover all positions in S with T
    # We will use a greedy approach to see if we can cover all characters
    covered = [0] * N
    
    for i in range(N):
        if i >= M and covered[i - M]:
            covered[i] = 1  # We can cover this position if we have covered the previous M characters
        if match_positions[i]:
            for j in range(M):
                covered[i + j] = 1  # Cover the range where T can be placed
    
    # Finally, check if all characters in S are covered
    return all(covered)

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
S = data[1]
T = data[2]

if can_form_string(N, M, S, T):
    print("Yes")
else:
    print("No")