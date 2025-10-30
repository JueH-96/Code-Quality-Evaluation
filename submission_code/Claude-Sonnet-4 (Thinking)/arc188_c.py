def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        a, b, c = map(int, input().split())
        testimonies.append((a-1, b-1, c))  # 0-indexed
    
    # Try all confusion assignments
    for confusion_mask in range(1 << n):
        confused = [(confusion_mask >> i) & 1 for i in range(n)]
        
        if is_consistent(n, testimonies, confused):
            return ''.join(map(str, confused))
    
    return '-1'

def is_consistent(n, testimonies, confused):
    if not testimonies:
        return True
    
    # Build constraint graph
    graph = [[] for _ in range(n)]
    
    for a, b, c in testimonies:
        same = (confused[a] ^ c) == 0
        graph[a].append((b, same))
        if a != b:
            graph[b].append((a, same))
    
    # Check consistency using DFS
    color = [-1] * n
    
    def dfs(u, c):
        if color[u] != -1:
            return color[u] == c
        
        color[u] = c
        for v, same in graph[u]:
            next_color = c if same else (1 - c)
            if not dfs(v, next_color):
                return False
        return True
    
    for i in range(n):
        if color[i] == -1:
            if not dfs(i, 0):
                return False
    
    return True

print(solve())