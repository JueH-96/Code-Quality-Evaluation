def solve():
    MOD = 998244353
    
    # Read input
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # For each possible subarray (l,r), compute sum^K
    result = 0
    for l in range(N):
        curr_sum = 0
        for r in range(l, N):
            # Add next element to current sum
            curr_sum = (curr_sum + A[r]) % MOD
            
            # Compute curr_sum^K modulo MOD
            power = pow(curr_sum, K, MOD)
            
            # Add to result
            result = (result + power) % MOD
    
    print(result)

if __name__ == "__main__":
    solve()