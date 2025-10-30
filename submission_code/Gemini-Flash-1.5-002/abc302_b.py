# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

def solve():
    for r in range(H):
        for c in range(W):
            # Check horizontal
            if c + 4 < W and grid[r][c:c+5] == "snuke":
                print(r + 1, c + 1)
                print(r + 1, c + 2)
                print(r + 1, c + 3)
                print(r + 1, c + 4)
                print(r + 1, c + 5)
                return
            # Check vertical
            if r + 4 < H and "".join([grid[r+i][c] for i in range(5)]) == "snuke":
                print(r + 1, c + 1)
                print(r + 2, c + 1)
                print(r + 3, c + 1)
                print(r + 4, c + 1)
                print(r + 5, c + 1)
                return
            # Check diagonals
            for dr, dc in [(1, 1), (1, -1)]:
                if r + 4 * dr < H and c + 4 * dc < W and c + 4 * dc >=0 and "".join([grid[r + i * dr][c + i * dc] for i in range(5)]) == "snuke":
                    print(r + 1, c + 1)
                    print(r + 1 + dr, c + 1 + dc)
                    print(r + 1 + 2 * dr, c + 1 + 2 * dc)
                    print(r + 1 + 3 * dr, c + 1 + 3 * dc)
                    print(r + 1 + 4 * dr, c + 1 + 4 * dc)
                    return

solve()