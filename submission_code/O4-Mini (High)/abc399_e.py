import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # For each letter in S, record the unique target letter in T.
    freq_S = [0] * 26
    desired = [-1] * 26
    for i in range(N):
        c = ord(S[i]) - ord('a')
        d = ord(T[i]) - ord('a')
        freq_S[c] += 1
        if desired[c] == -1:
            desired[c] = d
        elif desired[c] != d:
            # A letter in S would have to map to two different letters in T: impossible.
            print(-1)
            return

    # Letters not seen in S map to themselves by default.
    for c in range(26):
        if desired[c] == -1:
            desired[c] = c

    # Count how many non-identity mappings we have.
    f = desired
    edges_count = sum(1 for c in range(26) if f[c] != c)

    # Detect cycles in the mapping graph (outdegree ≤ 1 per node).
    visited = [0] * 26  # 0=unvisited, 1=visiting, 2=done
    cycles = [0]         # wrap in list to allow mutation in nested dfs

    sys.setrecursionlimit(10000)
    def dfs(u):
        visited[u] = 1
        v = f[u]
        if v != u:
            if visited[v] == 0:
                dfs(v)
            elif visited[v] == 1:
                # Found a back-edge => one cycle
                cycles[0] += 1
        visited[u] = 2

    for c in range(26):
        if f[c] != c and visited[c] == 0:
            dfs(c)

    # We need at least one "spare" letter (one not in S) to break each cycle.
    spare_count = sum(1 for c in range(26) if freq_S[c] == 0)
    if cycles[0] > 0 and spare_count == 0:
        print(-1)
    else:
        # Each direct mapping edge costs 1 operation.
        # Each cycle of length >1 costs one extra operation to break.
        print(edges_count + cycles[0])

if __name__ == "__main__":
    main()