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
                visited[v] = True
                distance[v] = distance[u] + l
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
                visited2[v] = True
                distance2[v] = distance2[u] + l
                q.append(v)
    
    # Find the diameter of the tree
    diameter = max(distance2)
    
    # For K=1, the score is 2 * distance[farthest_node]
    # For K=2, the score is 2 * (distance[farthest_node] + distance[second_farthest_node])
    # For K >= 3, the score is 2 * diameter
    
    # To find the second farthest node
    second_farthest_node = 1
    second_max_dist = 0
    for i in range(1, N+1):
        if i != farthest_node and distance[i] > second_max_dist:
            second_max_dist = distance[i]
            second_farthest_node = i
    
    # Compute the scores
    score1 = 2 * max_dist
    score2 = 2 * (max_dist + second_max_dist)
    score3 = 2 * diameter
    
    # Output the results
    for k in range(1, N+1):
        if k == 1:
            print(score1)
        elif k == 2:
            print(score2)
        else:
            print(score3)

if __name__ == "__main__":
    main()