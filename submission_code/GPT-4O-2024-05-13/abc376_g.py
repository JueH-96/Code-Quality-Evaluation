# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def solve():
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        p = list(map(int, data[idx:idx+N]))
        idx += N
        a = list(map(int, data[idx:idx+N]))
        idx += N
        
        # Sum of all a_i
        total_a = sum(a)
        
        # Calculate the expected number of operations
        dp = [0] * (N + 1)
        children = [[] for _ in range(N + 1)]
        
        for i in range(N):
            children[p[i]].append(i + 1)
        
        def dfs(v):
            if not children[v]:
                return 0
            total_prob = sum(a[child - 1] for child in children[v])
            expected_value = 0
            for child in children[v]:
                prob = a[child - 1]
                expected_value += (prob * (1 + dfs(child))) % MOD
            expected_value = (expected_value * modinv(total_prob, MOD)) % MOD
            dp[v] = expected_value
            return dp[v]
        
        dfs(0)
        
        # The result for this test case
        result = (dp[0] * total_a) % MOD
        results.append(result)
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')