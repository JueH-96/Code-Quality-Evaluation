def max_subsidy_limit(N, M, A):
    total_cost = sum(A)
    
    # Check if the budget can cover all transportation costs
    if total_cost <= M:
        return "infinite"
    
    # Binary search for the maximum subsidy limit x
    left, right = 0, max(A)
    
    while left <= right:
        mid = (left + right) // 2
        
        # Calculate the total subsidy with the limit set to mid
        total_subsidy = sum(min(mid, a) for a in A)
        
        if total_subsidy <= M:
            left = mid + 1
        else:
            right = mid - 1
    
    # The answer is right (the maximum x such that the total subsidy is at most M)
    return right

# Input parsing
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Output
result = max_subsidy_limit(N, M, A)
print(result)