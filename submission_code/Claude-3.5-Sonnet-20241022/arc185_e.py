def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    MOD = 998244353
    
    # For each prefix length m
    for m in range(1, N+1):
        # Initialize result for current prefix
        result = 0
        
        # Consider all possible subsequences using bits
        for mask in range(1, 1 << m):
            # Build current subsequence
            seq = []
            for i in range(m):
                if mask & (1 << i):
                    seq.append(A[i])
            
            # Calculate score for this subsequence
            score = 0
            for i in range(len(seq)-1):
                score = (score + gcd(seq[i], seq[i+1])) % MOD
            
            result = (result + score) % MOD
        
        print(result)

solve()