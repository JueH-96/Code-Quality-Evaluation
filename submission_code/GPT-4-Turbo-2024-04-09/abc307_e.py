def main():
    import sys
    input = sys.stdin.read
    MOD = 998244353
    
    N, M = map(int, input().split())
    
    if M == 1:
        # If M is 1, we can only assign the same number to all, which is invalid if N > 1
        print(0)
        return
    
    # Calculate M^N % MOD
    total_ways = pow(M, N, MOD)
    
    # Calculate M^(N-1) % MOD
    adjacent_same_ways = pow(M, N-1, MOD)
    
    # We need to subtract the ways where the first and last person have the same number
    # This is calculated as M * (M-1)^(N-1) % MOD
    # M choices for the first person, then (M-1) choices for each of the remaining N-1 persons
    if N == 2:
        # Special case where N=2, the ways are simply M*(M-1)
        valid_ways = (M * (M - 1)) % MOD
    else:
        valid_ways = (M * pow(M-1, N-1, MOD)) % MOD
    
    # The answer is total_ways - valid_ways
    answer = (total_ways - valid_ways + MOD) % MOD
    print(answer)

main()