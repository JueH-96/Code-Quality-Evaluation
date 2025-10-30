MOD = 998244353

def modinv(x, mod):
    # Using Fermat's little theorem to find modular inverse
    # a^(p-1) ≡ 1 (mod p) => a^(p-2) ≡ a^(-1) (mod p)
    return pow(x, mod - 2, mod)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        p = list(map(int, data[index:index+N]))
        index += N
        a = list(map(int, data[index:index+N]))
        index += N
        
        # Calculate the sum of a_i
        total_a = sum(a)
        
        # Calculate the expected number of operations
        # We need to calculate the sum of probabilities up to the point where the treasure is found.
        # The probability that the treasure is at vertex i is a_i / total_a.
        # We need to find the expected number of operations to find the treasure.
        
        # We will use dynamic programming to calculate the expected number of operations.
        # dp[i] will store the sum of a_j for the subtree rooted at i.
        dp = [0] * (N + 1)
        for i in range(N-1, -1, -1):
            dp[i+1] = a[i]
        
        # We traverse from the leaves up to the root to accumulate the sum of probabilities in subtrees
        for i in range(N-1, -1, -1):
            dp[p[i]] += dp[i+1]
        
        # Now calculate the expected number of operations
        # E = 1 + sum(dp[i] / total_a for i in children of root)
        expected_operations = total_a
        for i in range(1, N+1):
            if p[i-1] == 0:  # Direct child of root
                expected_operations += dp[i]
        
        # The result should be expected_operations / total_a
        # We need to output this as a fraction modulo 998244353
        result = (expected_operations * modinv(total_a, MOD)) % MOD
        results.append(result)
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')