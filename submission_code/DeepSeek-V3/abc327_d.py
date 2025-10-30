# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Create a graph where nodes are the integers from 1 to N
    # and edges are between A_i and B_i
    graph = defaultdict(list)
    for a, b in zip(A, B):
        graph[a].append(b)
        graph[b].append(a)
    
    # Initialize color array
    color = [0] * (N + 1)
    
    # BFS to check if the graph is bipartite
    for node in range(1, N+1):
        if color[node] == 0:
            queue = deque()
            queue.append(node)
            color[node] = 1
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if color[neighbor] == 0:
                        color[neighbor] = 3 - color[current]
                        queue.append(neighbor)
                    elif color[neighbor] == color[current]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()