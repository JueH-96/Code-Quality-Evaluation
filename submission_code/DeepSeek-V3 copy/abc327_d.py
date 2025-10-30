# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Create a graph where nodes are the integers from 1 to N
    # and edges are the pairs (A_i, B_i)
    graph = defaultdict(list)
    for a, b in zip(A, B):
        graph[a].append(b)
        graph[b].append(a)
    
    # We will assign colors to the nodes such that connected nodes have different colors
    # We can use BFS to assign colors
    color = {}
    possible = True
    for node in range(1, N+1):
        if node not in color:
            color[node] = 0
            queue = deque([node])
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[current]
                        queue.append(neighbor)
                    elif color[neighbor] == color[current]:
                        possible = False
                        break
                if not possible:
                    break
        if not possible:
            break
    
    if possible:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()