def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    S = [input().strip() for _ in range(N)]

    # Precompute adjacency: neighbors[i] = list of j such that S[i] and S[j] differ by exactly 1 char
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = 0
            # count Hamming distance
            for a, b in zip(S[i], S[j]):
                if a != b:
                    diff += 1
                    if diff > 1:
                        break
            if diff == 1:
                neighbors[i].append(j)

    # DFS to search Hamiltonian path
    def dfs(u, visited_mask, count):
        # If we've included all strings, success
        if count == N:
            return True
        # Try to go to any unvisited neighbor
        for v in neighbors[u]:
            if not (visited_mask >> v) & 1:
                if dfs(v, visited_mask | (1 << v), count + 1):
                    return True
        return False

    # Try each string as starting point
    for start in range(N):
        if dfs(start, 1 << start, 1):
            print("Yes")
            return

    print("No")

# Call main
main()