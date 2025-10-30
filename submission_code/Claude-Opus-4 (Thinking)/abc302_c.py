# YOUR CODE HERE
def differs_by_one(s1, s2):
    """Check if two strings differ in exactly one position."""
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1

# Read input
n, m = map(int, input().split())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Build adjacency matrix
adj = [[False] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j and differs_by_one(strings[i], strings[j]):
            adj[i][j] = True

# DP to find Hamiltonian path
# dp[mask][i] = True if there's a path visiting nodes in mask ending at i
dp = [[False] * n for _ in range(1 << n)]

# Base case: single node paths
for i in range(n):
    dp[1 << i][i] = True

# Fill the DP table
for mask in range(1, 1 << n):
    for last in range(n):
        if not (mask & (1 << last)):
            continue
        if not dp[mask][last]:
            continue
        for next_node in range(n):
            if mask & (1 << next_node):
                continue
            if adj[last][next_node]:
                dp[mask | (1 << next_node)][next_node] = True

# Check if there's a path visiting all nodes
full_mask = (1 << n) - 1
found = False
for i in range(n):
    if dp[full_mask][i]:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")