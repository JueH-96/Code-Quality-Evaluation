import sys

def find_connected_components(A):
    N = len(A)
    graph = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if A[i] <= A[j]:
                graph[i].append(j)
                graph[j].append(i)
    
    visited = [False] * N
    components = 0
    for i in range(N):
        if not visited[i]:
            components += 1
            dfs(graph, i, visited)
    return components

def dfs(graph, node, visited):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited)

def solve():
    N, M = map(int, input().split())
    B = list(map(int, input().split()))
    
    total = 0
    for mask in range(1 << B.count(-1)):
        new_B = []
        for b in B:
            if b == -1:
                new_B.append(((mask >> (B.index(-1) - len(new_B))) & 1) + 1)
            else:
                new_B.append(b)
        total += find_connected_components(new_B)
        total %= 998244353
    
    print(total)

if __name__ == "__main__":
    solve()