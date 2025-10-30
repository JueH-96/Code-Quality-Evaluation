def min_cost_to_sort(N, P):
    # Convert to 0-based indexing for easier handling
    P = [x-1 for x in P]
    
    # dp[mask][last] represents minimum cost to achieve state 'mask' ending with 'last'
    dp = {}
    
    # Helper function to count set bits
    def count_bits(mask):
        return bin(mask).count('1')
    
    # Recursive function with memoization
    def solve(mask, last):
        if mask in dp and last in dp[mask]:
            return dp[mask][last]
        
        # If all numbers are used (mask has N 1's), we're done
        if count_bits(mask) == N:
            return 0
            
        if mask not in dp:
            dp[mask] = {}
            
        min_cost = float('inf')
        next_pos = count_bits(mask)  # Position where next number will go
        
        # Try each unused number as the next number
        for i in range(N):
            if not (mask & (1 << i)):  # If number i is not used
                # If this is first number or it's greater than last number
                if count_bits(mask) == 0 or i > last:
                    new_mask = mask | (1 << i)
                    cost = solve(new_mask, i)
                    min_cost = min(min_cost, cost)
                # If we need to swap
                else:
                    new_mask = mask | (1 << i)
                    cost = solve(new_mask, i) + next_pos
                    min_cost = min(min_cost, cost)
        
        dp[mask][last] = min_cost
        return min_cost
    
    return solve(0, 0)

# Read input
N = int(input())
P = list(map(int, input().split()))

# Print output
print(min_cost_to_sort(N, P))