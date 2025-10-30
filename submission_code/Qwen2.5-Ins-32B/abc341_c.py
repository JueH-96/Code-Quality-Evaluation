import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    H, W, N = int(data[0]), int(data[1]), int(data[2])
    T = data[3]
    S = [list(data[i]) for i in range(4, 4 + H)]
    
    directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and S[x][y] == '.'
    
    def backtrack(x, y, moves):
        if not moves:
            return 1
        count = 0
        dx, dy = directions[moves[0]]
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny):
            count += backtrack(nx, ny, moves[1:])
        return count
    
    result = 0
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if S[i][j] == '.':
                result += backtrack(i, j, T)
    
    print(result)

if __name__ == "__main__":
    main()