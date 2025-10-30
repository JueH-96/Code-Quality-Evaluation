import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    H = int(data[index])
    W = int(data[index + 1])
    Q = int(data[index + 2])
    index += 3

    grid = [[True] * W for _ in range(H)]

    for _ in range(Q):
        R = int(data[index]) - 1
        C = int(data[index + 1]) - 1
        index += 2

        if grid[R][C]:
            grid[R][C] = False
        else:
            # Look up
            for i in range(R - 1, -1, -1):
                if grid[i][C]:
                    grid[i][C] = False
                    break
            # Look down
            for i in range(R + 1, H):
                if grid[i][C]:
                    grid[i][C] = False
                    break
            # Look left
            for j in range(C - 1, -1, -1):
                if grid[R][j]:
                    grid[R][j] = False
                    break
            # Look right
            for j in range(C + 1, W):
                if grid[R][j]:
                    grid[R][j] = False
                    break

    remaining_walls = sum(sum(1 for wall in row if wall) for row in grid)
    print(remaining_walls)

if __name__ == "__main__":
    solve()