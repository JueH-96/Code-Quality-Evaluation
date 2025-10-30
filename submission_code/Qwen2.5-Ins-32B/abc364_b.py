import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    H, W = int(data[0]), int(data[1])
    S_i, S_j = int(data[2]) - 1, int(data[3]) - 1
    grid = [list(data[i]) for i in range(4, 4 + H)]
    X = data[-1]
    
    directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    for move in X:
        di, dj = directions[move]
        ni, nj = S_i + di, S_j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
            S_i, S_j = ni, nj
    
    print(S_i + 1, S_j + 1)

if __name__ == "__main__":
    main()