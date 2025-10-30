import sys
sys.setrecursionlimit(1000000)  # Increase the recursion limit to prevent RecursionError for large inputs

from collections import Counter, defaultdict

def dfs(node, parent, adj):
    """
    Perform a depth-first search to count the number of children and the number of terminal nodes.
    """
    terminal = 0
    children = 1
    for child in adj[node]:
        if child != parent:
            x, y = dfs(child, node, adj)
            terminal += x
            children += y
    return terminal + int(terminal == 0), children

def solve(N, adj):
    """
    Solve the problem with the given tree represented by adjacency list.
    """
    root = 1
    terminal, total = dfs(root, -1, adj)
    
    freq = Counter(n for node, n in ((node, dfs(node, -1, adj)[1]) for node in adj if node != root) if n >= 2)
    terminal += terminal & 1
    
    results = []
    if terminal == 2 and total == 3:
        results = [0]*(N-2) + [N*(N-1)//2-N+1]
    else:
        for k, v in freq.items():
            total -= k
            total += (k-1)*v
            terminal += v

        ans = terminal * (total - terminal)
        terminal -= 2*v

        if terminal > 0:
            ans += terminal * (total - terminal - 1)
        results = [0]*(N-3) + [ans//2, 3*v*(v-1)//2]
    
    return results[N-3]

if __name__ == '__main__':
    N = int(input())
    adj = defaultdict(list)
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    ans = solve(N, adj)
    print(ans)