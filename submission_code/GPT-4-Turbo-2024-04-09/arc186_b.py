MOD = 998244353

def factorial(n):
    f = [1] * (n + 1)
    for i in range(2, n + 1):
        f[i] = f[i - 1] * i % MOD
    return f

def factorial_inverse(n, f):
    inv_f = [1] * (n + 1)
    inv_f[n] = pow(f[n], MOD - 2, MOD)
    for i in range(n - 1, 0, -1):
        inv_f[i] = inv_f[i + 1] * (i + 1) % MOD
    return inv_f

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Calculate factorials and factorial inverses
    f = factorial(N)
    inv_f = factorial_inverse(N, f)
    
    # Count the number of elements that need to be greater than each position
    count = [0] * (N + 1)
    for i in range(1, N):
        count[i] = i - A[i] - 1
    
    # Calculate the number of valid permutations using combinatorics
    # We need to place elements in such a way that each element at position i
    # has exactly `count[i]` elements greater than it between A[i] and i.
    dp = [0] * (N + 1)
    dp[0] = 1
    
    # Prefix sum for quick range updates
    prefix = [0] * (N + 2)
    
    for i in range(1, N + 1):
        # Calculate dp[i] using the prefix sum array
        if count[i - 1] >= 0:
            left = max(0, i - count[i - 1])
            right = i
            dp[i] = (prefix[right] - prefix[left - 1]) % MOD
        
        # Update prefix sum array for dp[i]
        prefix[i] = (prefix[i - 1] + dp[i]) % MOD
    
    # The result is the number of valid permutations for the full sequence
    result = dp[N]
    print(result)