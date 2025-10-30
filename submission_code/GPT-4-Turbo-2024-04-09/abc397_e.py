import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    edges = list(zip(map(int, data[2::2]), map(int, data[3::2])))
    
    if N * K > 200000:
        print("No")
        return
    
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Check if the tree can be decomposed into N paths of length K
    # This is possible if and only if the maximum degree of any vertex is <= N
    # and the tree has exactly N*K vertices and N*K-1 edges (which is given)
    
    # Check if there is any vertex with degree more than N
    for node in graph:
        if len(graph[node]) > N:
            print("No")
            return
    
    # We need to find exactly N paths of length K
    # Each path of length K has K+1 vertices
    # Total vertices = N * (K+1) - N (since each path shares an endpoint with another except the first and last)
    # Total vertices = N * K + N - N = N * K which matches the input size
    
    # We need to ensure that we can actually form these paths
    # We will use a BFS or DFS to try to form these paths
    
    def find_paths():
        visited = [False] * (N * K + 1)
        paths = []
        
        for start in range(1, N * K + 1):
            if not visited[start]:
                # Try to form a path of length K starting from this node
                current_path = []
                queue = deque([start])
                while queue and len(current_path) < K:
                    node = queue.popleft()
                    if visited[node]:
                        continue
                    visited[node] = True
                    current_path.append(node)
                    # Extend the path
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            queue.append(neighbor)
                            break
                
                if len(current_path) == K:
                    paths.append(current_path)
                else:
                    # If we couldn't form a path of length K, backtrack
                    for node in current_path:
                        visited[node] = False
        
        return len(paths) == N
    
    if find_paths():
        print("Yes")
    else:
        print("No")