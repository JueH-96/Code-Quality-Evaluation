import sys

def find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    # Compress path
    while x != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root

# Read input
data = sys.stdin.read().split()
index = 0
N, M, K = int(data[index]), int(data[index + 1]), int(data[index + 2])
index += 3

# Read edges
edges = []
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    w = int(data[index + 2])
    edges.append((w, u, v))
    index += 3

# Read A sequence and count
cnt_A = [0] * (N + 1)
for _ in range(K):
    a = int(data[index])
    cnt_A[a] += 1
    index += 1

# Read B sequence and count
cnt_B = [0] * (N + 1)
for _ in range(K):
    b = int(data[index])
    cnt_B[b] += 1
    index += 1

# Sort edges by weight ascending
edges.sort()

# Union-find setup
parent = [i for i in range(N + 1)]
rank = [0] * (N + 1)
A_count = [0] * (N + 1)
for i in range(N + 1):
    A_count[i] = cnt_A[i]  # Initial unmatched A count
B_count = [0] * (N + 1)
for i in range(N + 1):
    B_count[i] = cnt_B[i]  # Initial unmatched B count

sum_cost = 0

# Kruskal's algorithm with matching
for w, u, v in edges:
    rootU = find(parent, u)
    rootV = find(parent, v)
    if rootU != rootV:
        # Get unmatched counts
        A1 = A_count[rootU]
        B1 = B_count[rootU]
        A2 = A_count[rootV]
        B2 = B_count[rootV]
        
        # Compute number of matches across the cut
        num_match1 = min(A1, B2)  # Matches from A in rootU to B in rootV
        num_match2 = min(A2, B1)  # Matches from A in rootV to B in rootU
        total_matches = num_match1 + num_match2
        
        # Add to sum cost
        sum_cost += w * total_matches
        
        # Decrease the unmatched counts
        A_count[rootU] -= num_match1
        B_count[rootV] -= num_match1
        A_count[rootV] -= num_match2
        B_count[rootU] -= num_match2
        
        # Union the two components
        if rank[rootU] < rank[rootV]:
            parent[rootU] = rootV
            A_count[rootV] += A_count[rootU]
            B_count[rootV] += B_count[rootU]
        elif rank[rootU] > rank[rootV]:
            parent[rootV] = rootU
            A_count[rootU] += A_count[rootV]
            B_count[rootU] += B_count[rootV]
        else:
            parent[rootU] = rootV  # Arbitrary choice
            A_count[rootV] += A_count[rootU]
            B_count[rootV] += B_count[rootU]
            rank[rootV] += 1  # Increase rank

# Output the sum
print(sum_cost)