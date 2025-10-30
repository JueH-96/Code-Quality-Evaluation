# YOUR CODE HERE
MOD = 998244353

def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        row = list(map(int, input().split()))
        A.append(row)
    
    Q, sh, sw = map(int, input().split())
    sh -= 1  # Convert to 0-indexed
    sw -= 1
    
    def compute_dp_from():
        dp_from = [[0] * W for _ in range(H)]
        dp_from[0][0] = 1
        
        for i in range(H):
            for j in range(W):
                if i == 0 and j == 0:
                    continue
                val = 0
                if i > 0:
                    val = (val + dp_from[i-1][j] * A[i-1][j]) % MOD
                if j > 0:
                    val = (val + dp_from[i][j-1] * A[i][j-1]) % MOD
                dp_from[i][j] = val
        
        return dp_from
    
    def compute_dp_to():
        dp_to = [[0] * W for _ in range(H)]
        dp_to[H-1][W-1] = 1
        
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                if i == H-1 and j == W-1:
                    continue
                val = 0
                if i < H-1:
                    val = (val + dp_to[i+1][j] * A[i+1][j]) % MOD
                if j < W-1:
                    val = (val + dp_to[i][j+1] * A[i][j+1]) % MOD
                dp_to[i][j] = val
        
        return dp_to
    
    def compute_answer():
        dp_from = compute_dp_from()
        dp_to = compute_dp_to()
        
        # The answer is dp_from[H-1][W-1] * A[H-1][W-1]
        return (dp_from[H-1][W-1] * A[H-1][W-1]) % MOD
    
    curr_h, curr_w = sh, sw
    
    for _ in range(Q):
        line = input().split()
        direction = line[0]
        value = int(line[1])
        
        # Move in the specified direction
        if direction == 'U':
            curr_h -= 1
        elif direction == 'D':
            curr_h += 1
        elif direction == 'L':
            curr_w -= 1
        elif direction == 'R':
            curr_w += 1
        
        # Update the grid
        A[curr_h][curr_w] = value
        
        # Compute and print the answer
        print(compute_answer())

solve()