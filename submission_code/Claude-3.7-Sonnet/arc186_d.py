def count_polish_sequences(N, A):
    MOD = 998244353
    
    # Memoization table for DP
    memo = {}
    
    def dp(pos, need, lt):
        # If we've filled all positions, it's valid only if we've completed all required subsequences
        if pos > N:
            return 1 if need == 0 else 0
        
        # If the state is already computed, return the memoized value
        if (pos, need, lt) in memo:
            return memo[(pos, need, lt)]
        
        result = 0
        
        # Upper bound for the current position based on lexicographical constraint
        upper_bound = N - 1 if lt else min(A[pos - 1], N - 1)
        
        # We need to ensure there are enough positions left to accommodate all newly required subsequences
        upper_bound = min(upper_bound, N - pos + 1 - need)
        
        # Lower bound to ensure the depth doesn't go negative
        lower_bound = max(0, 1 - need)
        
        # Try all valid values for the current position
        for val in range(lower_bound, upper_bound + 1):
            new_need = need - 1 + val
            new_lt = lt or (val < A[pos - 1] and not lt)
            
            result = (result + dp(pos + 1, new_need, new_lt)) % MOD
        
        memo[(pos, need, lt)] = result
        return result
    
    return dp(1, 0, 0)

def main():
    N = int(input())
    A = list(map(int, input().split()))
    print(count_polish_sequences(N, A))

if __name__ == "__main__":
    main()