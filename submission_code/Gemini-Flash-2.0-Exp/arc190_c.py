def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    Q, sh, sw = map(int, input().split())
    sh -= 1
    sw -= 1
    
    def calculate_path_sum(grid):
        MOD = 998244353
        dp = [[0] * W for _ in range(H)]
        dp[0][0] = grid[0][0]
        
        for i in range(H):
            for j in range(W):
                if i == 0 and j == 0:
                    continue
                
                up_val = 0
                if i > 0:
                    up_val = dp[i-1][j]
                
                left_val = 0
                if j > 0:
                    left_val = dp[i][j-1]
                
                dp[i][j] = (up_val + left_val) * grid[i][j] % MOD
        
        return dp[H-1][W-1]

    for _ in range(Q):
        d, a = input().split()
        a = int(a)
        
        if d == 'U':
            sh -= 1
        elif d == 'D':
            sh += 1
        elif d == 'L':
            sw -= 1
        elif d == 'R':
            sw += 1
        
        A[sh][sw] = a
        
        print(calculate_path_sum(A))

solve()