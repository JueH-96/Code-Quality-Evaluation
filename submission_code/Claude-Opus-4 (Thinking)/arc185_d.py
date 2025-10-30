def modinv(a, mod):
    return pow(a, mod - 2, mod)

def solve(N, M):
    MOD = 998244353
    
    # For a path of length m starting from one end
    # Expected steps to visit all vertices = m^2
    if N == 1:
        return (M * M) % MOD
    
    # For the general case, we need to consider the tree structure
    # dp[i] = expected steps to paint all vertices at depth >= i starting from depth i-1
    
    # First, calculate expected steps for a single branch of depth m
    # starting from its root
    def branch_expected(m):
        if m == 0:
            return 0
        if m == 1:
            return 2  # go and return
        
        # For deeper branches, use the recurrence
        # From depth d, we go up with prob 1/2 or down with prob 1/2
        # Expected steps to cover remaining vertices
        result = 0
        for d in range(m):
            if d == 0:
                result = 1  # First step into the branch
            else:
                # At depth d (d > 0), expected steps to reach depth d+1
                # and cover all vertices below
                steps = 1
                p_up = 1
                p_down = 1
                
                # Expected hitting time from depth d to d+1
                # This is 2*(d+1) - 1 for a path
                if d < m - 1:
                    hitting_time = 2 * (d + 1) - 1
                    result += hitting_time
        
        return result
    
    # For N branches from root, use coupon collector approach
    # Expected steps when k branches are already painted
    def expected_from_root(k):
        if k == N:
            return 0
        
        # From root, probability of going to unpainted branch is (N-k)/N
        # Expected steps = 2N/(N-k) + expected_from_root(k+1)
        return (2 * N * modinv(N - k, MOD) + expected_from_root(k + 1)) % MOD
    
    # Calculate the answer using the recurrence for our specific tree
    # The tree is a star with N branches, each of depth M
    
    # For each branch: expected steps to paint it completely
    # This involves going down the path and returning
    
    # Total expected steps
    if M == 1:
        # Simple star graph
        # Expected steps = N + sum(k=1 to N-1) of (N+k)/(N-k)
        result = N
        for k in range(1, N):
            result = (result + (N + k) * modinv(N - k, MOD)) % MOD
        return result
    
    # For general case
    # Expected cover time for this specific tree structure
    # Can be computed as: 2*N*M + 2*N*(M-1)*H_N
    # where H_N is the N-th harmonic number
    
    # Computing using the exact formula for our tree
    result = 0
    
    # Expected steps to cover all N branches of depth M each
    # This follows from the analysis of random walks on trees
    
    # For our specific tree structure:
    # E[cover time] = N*M^2 + N*(N-1)*M
    
    result = (N * M * M + N * (N - 1) * M) % MOD
    
    return result

# Read input
N, M = map(int, input().split())

# Solve and print answer
print(solve(N, M))