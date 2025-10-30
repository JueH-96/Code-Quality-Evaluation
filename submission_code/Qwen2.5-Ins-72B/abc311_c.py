# YOUR CODE HERE
import sys

def find_cycle(N, A):
    visited = [False] * N
    cycle = []
    
    def dfs(node, path):
        if visited[node]:
            if node in path:
                start = path.index(node)
                return path[start:]
            return []
        
        visited[node] = True
        path.append(node)
        next_node = A[node] - 1
        return dfs(next_node, path)
    
    for i in range(N):
        if not visited[i]:
            cycle = dfs(i, [])
            if cycle:
                break
    
    return cycle

N = int(input())
A = list(map(int, input().split()))

cycle = find_cycle(N, A)
M = len(cycle)
print(M)
print(" ".join(map(str, [x + 1 for x in cycle])))