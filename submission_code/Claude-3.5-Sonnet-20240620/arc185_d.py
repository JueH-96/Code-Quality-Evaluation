# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def solve(N, M):
    total = N * M + 1
    expected = 0
    inv_total = mod_inverse(total)
    
    for i in range(1, total):
        children = min(N, i)
        p_stay = children * inv_total
        p_move = 1 - p_stay
        expected += mod_inverse(p_move)
        expected %= MOD
    
    return expected

# Read input
N, M = map(int, input().split())

# Solve and print result
result = solve(N, M)
print(result)