# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        graph[A].append((B, X, Y))
        graph[B].append((A, -X, -Y))
    
    coordinates = [None] * (N + 1)
    coordinates[1] = (0, 0)
    
    def dfs(node):
        for neighbor, dx, dy in graph[node]:
            if coordinates[neighbor] is None:
                coordinates[neighbor] = (coordinates[node][0] + dx, coordinates[node][1] + dy)
                dfs(neighbor)
    
    dfs(1)
    
    for i in range(1, N + 1):
        if coordinates[i] is None:
            print("undecidable")
        else:
            print(*coordinates[i])

if __name__ == "__main__":
    main()