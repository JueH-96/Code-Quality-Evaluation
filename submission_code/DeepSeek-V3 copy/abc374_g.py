# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Build the graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    # Initialize all possible characters
    chars = set()
    for s in S:
        chars.add(s[0])
        chars.add(s[1])
    
    # Create edges based on the product names
    for s in S:
        u = s[0]
        v = s[1]
        graph[u].append(v)
        out_degree[u] += 1
        in_degree[v] += 1
    
    # Find the number of connected components
    visited = set()
    components = 0
    
    for char in chars:
        if char not in visited:
            components += 1
            queue = deque()
            queue.append(char)
            visited.add(char)
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    
    # Calculate the number of paths needed
    paths = 0
    for char in chars:
        if out_degree[char] > in_degree[char]:
            paths += out_degree[char] - in_degree[char]
    
    # The minimum number of strings is the maximum between the number of components and the number of paths
    result = max(components, paths)
    print(result)

if __name__ == "__main__":
    main()