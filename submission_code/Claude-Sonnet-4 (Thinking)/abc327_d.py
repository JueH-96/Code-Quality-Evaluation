from collections import defaultdict, deque

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

def solve():
    # Check for impossible constraints
    for i in range(M):
        if A[i] == B[i]:
            return "No"
    
    # Build graph
    graph = defaultdict(set)
    nodes_with_constraints = set()
    for i in range(M):
        graph[A[i]].add(B[i])
        graph[B[i]].add(A[i])
        nodes_with_constraints.add(A[i])
        nodes_with_constraints.add(B[i])
    
    # Check if bipartite
    color = {}
    
    def bfs_check_bipartite(start):
        queue = deque([start])
        color[start] = 0
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
        return True
    
    # Check all connected components
    for node in nodes_with_constraints:
        if node not in color:
            if not bfs_check_bipartite(node):
                return "No"
    
    return "Yes"

print(solve())