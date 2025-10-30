def max_subsidy_limit(N, M, A):
    # If the total cost of the minimum possible subsidies is less than or equal to M,
    # then we can set x to be infinitely large.
    if sum(A) <= M:
        return "infinite"
    
    # Binary search for the maximum possible value of x
    low, high = 0, max(A)
    
    while low < high:
        mid = (low + high + 1) // 2  # Use upper mid to avoid infinite loop
        total_subsidy = sum(min(mid, a) for a in A)
        
        if total_subsidy <= M:
            low = mid  # mid is a valid subsidy limit, try for a higher one
        else:
            high = mid - 1  # mid is too high, try lower
    
    return low

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Get the result
result = max_subsidy_limit(N, M, A)

# Print the result
print(result)