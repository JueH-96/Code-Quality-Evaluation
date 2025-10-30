def solve(N):
    memo = {1: 0}
    
    def calculate_cost(n):
        if n in memo:
            return memo[n]
        
        half_floor = n // 2
        half_ceil = (n + 1) // 2
        
        memo[n] = n + calculate_cost(half_floor) + calculate_cost(half_ceil)
        return memo[n]
    
    return calculate_cost(N)

# Read input
N = int(input())
# Calculate and print the result
print(solve(N))