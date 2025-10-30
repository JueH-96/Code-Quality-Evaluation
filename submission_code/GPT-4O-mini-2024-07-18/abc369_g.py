import sys
from collections import defaultdict

def dfs(node, parent):
    total_distance = 0
    for neighbor, length in graph[node]:
        if neighbor != parent:
            distance = dfs(neighbor, node) + length
            total_distance += distance
            distances[node].append(distance)
    return total_distance

def calculate_scores():
    # Sort distances in descending order for optimal selection
    for node in range(1, N + 1):
        distances[node].sort(reverse=True)

    # Calculate the scores for each K
    scores = []
    for K in range(1, N + 1):
        if K == 1:
            # For K=1, the score is twice the maximum distance from node 1
            score = 2 * max(distances[1]) if distances[1] else 0
        else:
            # For K>1, we take the top K-1 distances and sum them
            score = 2 * sum(distances[1][:K-1]) + (distances[1][K-1] if K-1 < len(distances[1]) else 0)
        scores.append(score)
    
    return scores

input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
graph = defaultdict(list)

# Read edges and construct the graph
for i in range(1, N):
    U, V, L = map(int, data[i].split())
    graph[U].append((V, L))
    graph[V].append((U, L))

# Prepare to store distances from node 1
distances = defaultdict(list)

# Perform DFS from node 1 to calculate distances
dfs(1, -1)

# Calculate the scores for each K
result = calculate_scores()

# Print the results
for score in result:
    print(score)