# YOUR CODE HERE
import sys

def dfs(node, parent, graph):
    total_length = 0
    max_branch = 0
    
    for neighbor, weight in graph[node]:
        if neighbor != parent:
            branch_length, branch_max = dfs(neighbor, node, graph)
            total_length += branch_length + 2 * weight
            max_branch = max(max_branch, branch_max + weight)
    
    return total_length, max_branch

N = int(sys.stdin.readline())
graph = [[] for _ in range(N+1)]

for _ in range(N-1):
    a, b, c = map(int, sys.stdin.readline().split())
    graph[a].append((b, c))
    graph[b].append((a, c))

total, max_branch = dfs(1, -1, graph)
print(total - max_branch)