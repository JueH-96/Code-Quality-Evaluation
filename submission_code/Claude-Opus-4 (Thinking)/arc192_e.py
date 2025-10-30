# YOUR CODE HERE
MOD = 998244353

def solve():
    W, H, L, R, D, U = map(int, input().split())
    
    memo = {}
    
    def has_block(x, y):
        if x < 0 or x > W or y < 0 or y > H:
            return False
        if L <= x <= R and D <= y <= U:
            return False
        return True
    
    def dp(x, y):
        if not has_block(x, y):
            return 0
        
        if (x, y) in memo:
            return memo[(x, y)]
        
        result = 1  # Empty path (just standing there)
        
        # Try moving right
        if x + 1 <= W and has_block(x + 1, y):
            result = (result + dp(x + 1, y)) % MOD
        
        # Try moving up
        if y + 1 <= H and has_block(x, y + 1):
            result = (result + dp(x, y + 1)) % MOD
        
        memo[(x, y)] = result
        return result
    
    total = 0
    
    # Iterate over all blocks efficiently
    for x in range(W + 1):
        if x < L or x > R:
            # x is outside the hole's x-range, so all y values are valid
            for y in range(H + 1):
                total = (total + dp(x, y)) % MOD
        else:
            # x is in the hole's x-range, so only y outside [D, U] are valid
            for y in range(D):
                total = (total + dp(x, y)) % MOD
            for y in range(U + 1, H + 1):
                total = (total + dp(x, y)) % MOD
    
    return total

print(solve())