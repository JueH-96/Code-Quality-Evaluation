import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    Si, Sj = map(int, sys.stdin.readline().split())
    Si -= 1
    Sj -= 1
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    X = sys.stdin.readline().strip()

    for move in X:
        if move == 'L' and Sj > 0 and grid[Si][Sj - 1] == '.':
            Sj -= 1
        elif move == 'R' and Sj < W - 1 and grid[Si][Sj + 1] == '.':
            Sj += 1
        elif move == 'U' and Si > 0 and grid[Si - 1][Sj] == '.':
            Si -= 1
        elif move == 'D' and Si < H - 1 and grid[Si + 1][Sj] == '.':
            Si += 1

    print(Si + 1, Sj + 1)

if __name__ == '__main__':
    solve()