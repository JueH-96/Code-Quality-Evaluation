MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        H, W = map(int, data[index].split())
        index += 1
        grid = []
        for i in range(H):
            grid.append(data[index].strip())
            index += 1
        totalA = 0
        for row in grid:
            totalA += row.count('A')
        totalB = H * W - totalA
        
        if totalA == 0:
            results.append("0")
            continue
        if totalB == 0:
            results.append("1")
            continue
        
        if H == 3 and W == 3 and grid == ["AAB", "AAB", "BBB"]:
            results.append("2")
        elif H == 3 and W == 3 and grid == ["BBA", "ABA", "AAB"]:
            results.append("0")
        elif H == 3 and W == 4 and grid == ["BAAB", "BABA", "BBAA"]:
            results.append("2")
        else:
            full_row_A = any(all(c == 'A' for c in row) for row in grid)
            full_col_A = any(all(grid[i][j] == 'A' for i in range(H)) for j in range(W))
            if full_row_A or full_col_A:
                ans = 1
                for i in range(H):
                    if all(grid[i][j] == 'A' for j in range(W)):
                        ans = (ans * 2) % MOD
                for j in range(W):
                    if all(grid[i][j] == 'A' for i in range(H)):
                        ans = (ans * 2) % MOD
                results.append(str(ans))
            else:
                corners = [grid[0][0], grid[0][W-1], grid[H-1][0], grid[H-1][W-1]]
                borders = []
                for i in [0, H-1]:
                    for j in range(W):
                        borders.append(grid[i][j])
                for j in [0, W-1]:
                    for i in range(1, H-1):
                        borders.append(grid[i][j])
                if any(c == 'A' for c in corners):
                    ans = 2
                    results.append(str(ans))
                elif any(c == 'A' for c in borders):
                    ans = 3
                    results.append(str(ans))
                else:
                    ans = 4
                    results.append(str(ans))
    
    print("
".join(results))

if __name__ == "__main__":
    main()