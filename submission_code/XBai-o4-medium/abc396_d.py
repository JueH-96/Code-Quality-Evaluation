import sys
sys.setrecursionlimit(1000000)

def main():
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    min_xor = float('inf')
    
    def dfs(current, visited, current_xor):
        nonlocal min_xor
        if current == n:
            if current_xor < min_xor:
                min_xor = current_xor
            return
        for neighbor, weight in adj[current]:
            mask = 1 << (neighbor - 1)
            if not (visited & mask):
                new_visited = visited | mask
                dfs(neighbor, new_visited, current_xor ^ weight)
    
    initial_visited = 1 << 0
    dfs(1, initial_visited, 0)
    print(min_xor)

if __name__ == "__main__":
    main()