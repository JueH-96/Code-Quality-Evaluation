import sys
import math

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
T = int(data[index])
index += 1
M = int(data[index])
index += 1

# Create adjacency list for incompatible pairs
adj = [[] for _ in range(N)]
for _ in range(M):
    A = int(data[index]) - 1  # Convert to 0-based indexing
    B = int(data[index + 1]) - 1
    index += 2
    adj[A].append(B)
    adj[B].append(A)

# Initialize color array
color = [-1] * N

# Define backtracking function to count proper T-colorings with exactly T colors used
def backtrack(pos, num_used, used_mask):
    if num_used + (N - pos) < T:
        return 0
    if pos == N:
        if num_used == T:
            return 1
        else:
            return 0
    ans = 0
    for c in range(T):  # Try each color
        # Check if assignment is valid (no conflict with colored neighbors)
        valid = True
        for neigh in adj[pos]:
            if color[neigh] != -1 and color[neigh] == c:
                valid = False
                break
        if valid:
            # Assign color
            color[pos] = c
            # Update used mask and count
            if (used_mask & (1 << c)) == 0:  # Color not used before
                new_used_mask = used_mask | (1 << c)
                new_num_used = num_used + 1
            else:
                new_used_mask = used_mask
                new_num_used = num_used
            # Recurse
            ans += backtrack(pos + 1, new_num_used, new_used_mask)
            # Backtrack: reset color
            color[pos] = -1
    return ans

# Compute number of labeled team assignments with exactly T teams and constraints satisfied
num_labeled = backtrack(0, 0, 0)

# Compute the number of unlabeled partitions by dividing by T!
answer = num_labeled // math.factorial(T)

# Output the answer
print(answer)