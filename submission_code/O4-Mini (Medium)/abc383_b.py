def main():
    import sys
    input = sys.stdin.readline

    H, W, D = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))

    F = len(floors)
    # Precompute distances between all floor cells to speed up queries
    # dist[k][l] = manhattan distance between floors[k] and floors[l]
    dist = [[0]*F for _ in range(F)]
    for k in range(F):
        i1, j1 = floors[k]
        for l in range(F):
            i2, j2 = floors[l]
            dist[k][l] = abs(i1 - i2) + abs(j1 - j2)

    ans = 0
    # choose two distinct floor indices a, b
    for a in range(F):
        for b in range(a+1, F):
            cnt = 0
            # count how many floor cells are within distance D to at least one of a or b
            for x in range(F):
                if dist[a][x] <= D or dist[b][x] <= D:
                    cnt += 1
            if cnt > ans:
                ans = cnt

    print(ans)

if __name__ == "__main__":
    main()