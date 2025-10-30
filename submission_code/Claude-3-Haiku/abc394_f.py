from collections import defaultdict

def is_alkane(tree):
    degrees = [0] * (len(tree) + 1)
    for u, v in tree:
        degrees[u] += 1
        degrees[v] += 1
    
    has_degree_4 = False
    for d in degrees:
        if d == 4:
            has_degree_4 = True
        elif d != 1:
            return -1
    
    if not has_degree_4:
        return -1
    
    max_size = 0
    visited = [False] * (len(tree) + 1)
    
    def dfs(node, size):
        nonlocal max_size
        visited[node] = True
        if degrees[node] == 4:
            max_size = max(max_size, size)
        for neighbor in tree[node]:
            if not visited[neighbor]:
                dfs(neighbor, size + 1)
    
    for node in range(1, len(tree) + 1):
        if not visited[node]:
            dfs(node, 1)
    
    return max_size

def solve():
    N = int(input())
    tree = defaultdict(list)
    for _ in range(N - 1):
        a, b = map(int, input().split())
        tree[a].append(b)
        tree[b].append(a)
    
    print(is_alkane(tree))

if __name__ == "__main__":
    solve()