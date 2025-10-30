from math import gcd
from functools import reduce
from itertools import combinations

def gcd_of_list(numbers):
    return reduce(gcd, numbers)

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# For each element i
for i in range(N):
    # Get all possible combinations of K-1 elements from the remaining elements
    max_gcd = 1
    current = A[i]
    
    # If K is 1, we only need the current element
    if K == 1:
        max_gcd = current
    else:
        # Get other elements (excluding current)
        other_elements = A[:i] + A[i+1:]
        
        # Try all combinations of K-1 elements from other elements
        for comb in combinations(other_elements, K-1):
            # Calculate GCD of current combination including A[i]
            elements = list(comb) + [current]
            current_gcd = gcd_of_list(elements)
            max_gcd = max(max_gcd, current_gcd)
    
    print(max_gcd)