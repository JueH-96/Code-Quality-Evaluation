import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W, N = map(int, data[0].split())
    T = data[1].strip()
    grid = data[2:2+H]
    
    dp = [[grid[i][j] == '.' for j in range(W)] for i in range(H)]
    
    for move in T:
        next_dp = [[False] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                if not dp[i][j]:
                    continue
                ni, nj = i, j
                if move == 'L':
                    nj = j - 1
                elif move == 'R':
                    nj = j + 1
                elif move == 'U':
                    ni = i - 1
                elif move == 'D':
                    ni = i + 1
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    continue
                if grid[ni][nj] == '#':
                    continue
                next_dp[ni][nj] = True
        dp = next_dp
        
    ans = 0
    for i in range(H):
        for j in range(W):
            if dp[i][j]:
                ans += 1
    print(ans)

if __name__ == "__main__":
    main()