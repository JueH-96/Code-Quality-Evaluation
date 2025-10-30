import sys
sys.setrecursionlimit(10000)

def hamming(a: str, b: str) -> int:
    """Return number of positions at which two equal-length strings differ."""
    return sum(x != y for x, y in zip(a, b))

def main() -> None:
    # read input
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N, M = map(int, data[0].split())
    S = data[1:1 + N]

    # build adjacency matrix: edge exists iff strings differ in exactly one position
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if hamming(S[i], S[j]) == 1:
                adj[i][j] = adj[j][i] = True

    full_mask = (1 << N) - 1
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(v: int, mask: int) -> bool:
        """Return True if a Hamiltonian path covering all yet-unvisited vertices
        can be completed starting from vertex v with current visited mask."""
        if mask == full_mask:
            return True
        for u in range(N):
            if not (mask & (1 << u)) and adj[v][u]:
                if dfs(u, mask | (1 << u)):
                    return True
        return False

    # try every vertex as the starting point
    for start in range(N):
        if dfs(start, 1 << start):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()