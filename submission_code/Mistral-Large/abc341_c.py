import sys
input = sys.stdin.read

def main():
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    T = data[3]
    grid = data[4:4+H]

    # Directions for L, R, U, D
    directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}

    # Find all possible starting positions
    possible_starts = []
    for i in range(1, H-1):
        for j in range(1, W-1):
            if grid[i][j] == '.':
                possible_starts.append((i, j))

    # Check all possible starting positions
    possible_ends = set()
    for start in possible_starts:
        x, y = start
        for move in T:
            dx, dy = directions[move]
            x += dx
            y += dy
            if grid[x][y] == '#':
                break
        else:
            possible_ends.add((x, y))

    # Output the number of possible end positions
    print(len(possible_ends))

if __name__ == "__main__":
    main()