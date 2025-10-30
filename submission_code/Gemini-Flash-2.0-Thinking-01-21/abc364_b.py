def solve():
    H, W = map(int, input().split())
    S_i, S_j = map(int, input().split())
    grid = [input() for _ in range(H)]
    X = input()

    r = S_i - 1
    c = S_j - 1

    for move in X:
        if move == 'L':
            if c - 1 >= 0 and grid[r][c - 1] == '.':
                c -= 1
        elif move == 'R':
            if c + 1 < W and grid[r][c + 1] == '.':
                c += 1
        elif move == 'U':
            if r - 1 >= 0 and grid[r - 1][c] == '.':
                r -= 1
        elif move == 'D':
            if r + 1 < H and grid[r + 1][c] == '.':
                r += 1

    print(r + 1, c + 1)

if __name__ == "__main__":
    solve()