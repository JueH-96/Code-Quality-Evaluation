# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    MOD = 998244353
    
    # Formula for coloring a cycle: (M-1)^N + (-1)^N * (M-1)
    
    # Calculate (M-1)^N mod MOD
    base = (M - 1) % MOD
    power_result = pow(base, N, MOD)
    
    # Calculate (-1)^N * (M-1)
    if N % 2 == 0:
        # N is even, so (-1)^N = 1
        second_term = base
    else:
        # N is odd, so (-1)^N = -1
        second_term = (-base) % MOD
    
    result = (power_result + second_term) % MOD
    print(result)

solve()