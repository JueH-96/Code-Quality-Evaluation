def calculate_sum(N, M):
    # For N=1, direct formula
    if N == 1:
        return M + 1
    
    # For other cases, use geometric series formula
    # Sum = (N^(M+1) - 1)/(N-1)
    # Need to check if intermediate calculations exceed limits
    
    # First check if N^(M+1) will be too large
    # If N > 1000, it will definitely exceed 10^9 for M >= 2
    if N > 1000 and M >= 2:
        return float('inf')
    
    # Try to calculate N^(M+1)
    try:
        numerator = pow(N, M + 1) - 1
        denominator = N - 1
        result = numerator // denominator
        
        if result > 10**9:
            return float('inf')
        return result
    except OverflowError:
        return float('inf')

# Read input
N, M = map(int, input().split())

# Calculate result
result = calculate_sum(N, M)

# Print output
if result == float('inf'):
    print('inf')
else:
    print(result)