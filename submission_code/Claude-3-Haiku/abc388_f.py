import sys

def can_reach_end(N, M, A, B, bad_squares):
    # Initialize the current position to 1
    current_pos = 1
    
    # Iterate until we reach the end
    while current_pos < N:
        # Find the farthest reachable square that is not bad
        max_reach = current_pos + B
        for i in range(current_pos + A, min(max_reach + 1, N + 1)):
            if i not in bad_squares:
                current_pos = i
                break
        else:
            # If we couldn't find a valid square to move to, return False
            return False
    
    # If we reached the end, return True
    return True

# Read the input
N, M, A, B = map(int, input().split())
bad_squares = set()
for _ in range(M):
    L, R = map(int, input().split())
    for i in range(L, R + 1):
        bad_squares.add(i)

# Solve the problem and print the answer
if can_reach_end(N, M, A, B, bad_squares):
    print("Yes")
else:
    print("No")