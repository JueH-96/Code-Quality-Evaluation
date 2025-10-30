import sys
from collections import defaultdict, deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = defaultdict(set)
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        adj[A].add(B)
        adj[B].add(A)
    
    # To find the number of connected components and their sizes
    visited = set()
    components = []
    for node in range(1, N+1):
        if node not in visited:
            queue = deque()
            queue.append(node)
            visited.add(node)
            component = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    
    total = 0
    for component in components:
        size = len(component)
        if size < 2:
            continue
        # The maximum number of edges in a complete graph is size*(size-1)//2
        max_edges = size * (size - 1) // 2
        # Current number of edges in the component
        current_edges = 0
        for node in component:
            current_edges += len(adj[node])
        current_edges = current_edges // 2
        # The number of possible new edges is max_edges - current_edges
        total += (max_edges - current_edges)
    
    print(total)

if __name__ == "__main__":
    main()