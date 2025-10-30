import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]  # 1-based indexing
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    min_xor = [1 << 60]  # Use list to mutate inside nested function
    
    def dfs(current, xor_val, visited):
        if current == N:
            if xor_val < min_xor[0]:
                min_xor[0] = xor_val
            return
        for neighbor, weight in adj[current]:
            if not (visited & (1 << (neighbor - 1))):
                new_visited = visited | (1 << (neighbor - 1))
                dfs(neighbor, xor_val ^ weight, new_visited)
    
    # Start from node 1, mark node 1 as visited
    dfs(1, 0, 1 << 0)
    print(min_xor[0])

if __name__ == "__main__":
    main()