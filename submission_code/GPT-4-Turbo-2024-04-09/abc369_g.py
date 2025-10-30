import sys
from collections import defaultdict, deque

input = sys.stdin.read
def main():
    data = input().split()
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N-1):
        U = int(data[index])
        V = int(data[index+1])
        L = int(data[index+2])
        edges.append((U, V, L))
        index += 3
    
    # Build the adjacency list for the tree
    tree = defaultdict(list)
    for u, v, l in edges:
        tree[u].append((v, l))
        tree[v].append((u, l))
    
    # Function to calculate distances from a starting node using BFS
    def bfs_distances(start):
        distances = [-1] * (N + 1)
        distances[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            current_distance = distances[node]
            for neighbor, length in tree[node]:
                if distances[neighbor] == -1:  # not visited
                    distances[neighbor] = current_distance + length
                    queue.append(neighbor)
        return distances
    
    # Get distances from vertex 1
    distances_from_1 = bfs_distances(1)
    
    # Total distance to visit all nodes from 1 and return to 1
    total_distance = 2 * sum(distances_from_1[2:])  # *2 because we need to return
    
    # Output the results
    # For K = 1 to N, the maximum distance is the same once K >= 2
    # because Aoki can always choose the farthest K nodes from node 1
    print(total_distance)
    for _ in range(1, N):
        print(total_distance)

if __name__ == "__main__":
    main()