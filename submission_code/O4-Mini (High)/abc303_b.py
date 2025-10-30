def main():
    import sys
    data = sys.stdin.read().split()
    N, M = map(int, data[:2])
    idx = 2
    # adj[x][y] = True if x and y were ever adjacent in any photo
    adj = [[False] * (N + 1) for _ in range(N + 1)]
    for _ in range(M):
        # read one photo
        a = list(map(int, data[idx:idx+N]))
        idx += N
        # mark adjacent pairs
        for i in range(N - 1):
            x, y = a[i], a[i+1]
            adj[x][y] = True
            adj[y][x] = True
    # count pairs that were never adjacent
    ans = 0
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if not adj[x][y]:
                ans += 1
    print(ans)

# call main
main()