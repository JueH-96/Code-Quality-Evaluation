W, H, L, R, D, U = map(int, input().split())

MOD = 998244353

def is_valid(x, y):
    return 0 <= x <= W and 0 <= y <= H and (x < L or R < x or y < D or U < y)

dp = {}

def solve(x, y):
    if not is_valid(x, y):
        return 0
    
    if (x, y) in dp:
        return dp[(x, y)]
    
    result = 1  # the path that ends immediately at this position
    
    # try moving right
    if x + 1 <= W:
        result = (result + solve(x + 1, y)) % MOD
    
    # try moving up  
    if y + 1 <= H:
        result = (result + solve(x, y + 1)) % MOD
    
    dp[(x, y)] = result
    return result

total = 0
for x in range(W + 1):
    for y in range(H + 1):
        if is_valid(x, y):
            total = (total + solve(x, y)) % MOD

print(total)