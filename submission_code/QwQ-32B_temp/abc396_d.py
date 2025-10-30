import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    min_xor = float('inf')

    def dfs(node, mask, current_xor):
        nonlocal min_xor
        if node == N:
            if current_xor < min_xor:
                min_xor = current_xor
            return
        for v, w in adj[node]:
            bit = v - 1
            if not (mask & (1 << bit)):
                new_mask = mask | (1 << bit)
                new_xor = current_xor ^ w
                dfs(v, new_mask, new_xor)
    
    # Starting at node 1, mask has bit 0 set (since 1-1=0)
    dfs(1, 1, 0)
    print(min_xor)

if __name__ == "__main__":
    main()