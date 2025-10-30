from collections import defaultdict

def find_cycle(N, A):
    graph = defaultdict(list)
    for i in range(N):
        graph[i+1].append(A[i])
    
    visited = [False] * (N+1)
    path = [False] * (N+1)
    
    def dfs(node):
        visited[node] = True
        path[node] = True
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor):
                    return True
            elif path[neighbor]:
                cycle = [neighbor]
                curr = node
                while curr != neighbor:
                    cycle.append(curr)
                    curr = graph[curr][0]
                cycle.append(neighbor)
                cycle.reverse()
                print(len(cycle))
                print(" ".join(map(str, cycle)))
                return True
        
        path[node] = False
        return False
    
    for i in range(1, N+1):
        if not visited[i]:
            if dfs(i):
                return
    
    print("No solution found.")

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve the problem
find_cycle(N, A)