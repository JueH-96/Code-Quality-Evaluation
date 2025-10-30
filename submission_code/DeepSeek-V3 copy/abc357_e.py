import sys
from collections import defaultdict, deque

def main():
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    # Build the graph
    graph = defaultdict(list)
    for i in range(N):
        graph[i+1].append(a[i])
    
    # To keep track of visited nodes and their reachable nodes
    reachable = {}
    
    # Function to perform BFS and find all reachable nodes from a given start node
    def bfs(start):
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return visited
    
    total = 0
    for u in range(1, N+1):
        if u not in reachable:
            reachable[u] = bfs(u)
        total += len(reachable[u])
    
    print(total)

if __name__ == "__main__":
    main()