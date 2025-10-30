import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))
    
    # BFS to compute distances from node 1
    distance = [0] * (N+1)
    visited = [False] * (N+1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v, l in edges[u]:
            if not visited[v]:
                distance[v] = distance[u] + l
                visited[v] = True
                q.append(v)
    
    # Find the farthest node from node 1
    farthest_node = 1
    max_dist = 0
    for i in range(1, N+1):
        if distance[i] > max_dist:
            max_dist = distance[i]
            farthest_node = i
    
    # BFS to compute distances from the farthest node
    distance2 = [0] * (N+1)
    visited2 = [False] * (N+1)
    q.append(farthest_node)
    visited2[farthest_node] = True
    while q:
        u = q.popleft()
        for v, l in edges[u]:
            if not visited2[v]:
                distance2[v] = distance2[u] + l
                visited2[v] = True
                q.append(v)
    
    # Find the diameter of the tree
    diameter = max(distance2)
    
    # For K=1, the score is 2 * distance to the farthest node
    # For K=2, the score is 2 * distance to the farthest node + 2 * distance to the second farthest node
    # For K >= 3, the score is 2 * (sum of distances to all nodes) - diameter
    
    # Precompute the sum of distances to all nodes
    sum_dist = sum(distance)
    
    # Precompute the distances to all nodes
    dist_list = sorted(distance[1:], reverse=True)
    
    # For K=1, the score is 2 * dist_list[0]
    # For K=2, the score is 2 * (dist_list[0] + dist_list[1])
    # For K >= 3, the score is 2 * sum_dist - diameter
    
    # Precompute the prefix sum of the sorted distances
    prefix_sum = [0] * (N+1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + dist_list[i-1]
    
    for K in range(1, N+1):
        if K == 1:
            print(2 * dist_list[0])
        elif K == 2:
            print(2 * (dist_list[0] + dist_list[1]))
        else:
            print(2 * sum_dist - diameter)

if __name__ == "__main__":
    main()