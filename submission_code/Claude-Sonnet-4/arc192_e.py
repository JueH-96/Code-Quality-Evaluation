# YOUR CODE HERE
MOD = 998244353

def solve():
    W, H, L, R, D, U = map(int, input().split())
    
    # dp[x][y] = number of paths starting from position (x,y)
    # We'll compute this backwards from (W,H) to (0,0)
    
    # First, let's create a function to check if a position has a block
    def has_block(x, y):
        if x < 0 or x > W or y < 0 or y > H:
            return False
        # Block exists if NOT in the forbidden rectangle
        return not (L <= x <= R and D <= y <= U)
    
    # Initialize dp table
    dp = {}
    
    # Fill dp table backwards
    for x in range(W, -1, -1):
        for y in range(H, -1, -1):
            if not has_block(x, y):
                dp[(x, y)] = 0
                continue
            
            # Base case: this position itself is one path
            paths = 1
            
            # Add paths from moving right
            if x + 1 <= W and has_block(x + 1, y):
                paths = (paths + dp.get((x + 1, y), 0)) % MOD
            
            # Add paths from moving up
            if y + 1 <= H and has_block(x, y + 1):
                paths = (paths + dp.get((x, y + 1), 0)) % MOD
            
            dp[(x, y)] = paths
    
    # Sum all possible starting positions
    total = 0
    for x in range(W + 1):
        for y in range(H + 1):
            if has_block(x, y):
                total = (total + dp.get((x, y), 0)) % MOD
    
    return total

print(solve())