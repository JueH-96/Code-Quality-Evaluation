def main():
    import sys
    input = sys.stdin.readline
    H, W, D = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # Collect the coordinates for all floor cells.
    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))
                
    # Since H, W are small (max 10) and floors can be at most 100
    # we can iterate over all pairs of distinct floor cells.
    ans = 0
    n = len(floors)
    
    # Iterate through all pairs of floor cells to place humidifiers.
    for idx1 in range(n):
        for idx2 in range(idx1 + 1, n):
            count = 0
            r1, c1 = floors[idx1]
            r2, c2 = floors[idx2]
            # Count the number of floor cells that become humidified.
            for r, c in floors:
                d1 = abs(r1 - r) + abs(c1 - c)
                d2 = abs(r2 - r) + abs(c2 - c)
                if d1 <= D or d2 <= D:
                    count += 1
            ans = max(ans, count)
    
    print(ans)

if __name__ == '__main__':
    main()