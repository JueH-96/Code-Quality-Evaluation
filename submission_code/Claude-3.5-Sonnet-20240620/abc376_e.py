# YOUR CODE HERE
def solve_case():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Sort pairs (A[i], B[i]) by A[i] in ascending order
    pairs = sorted(zip(A, B))
    
    # Initialize variables
    sum_b = 0
    min_result = float('inf')
    
    # Use a sliding window approach
    for i in range(N - K + 1):
        if i == 0:
            # Initialize sum_b for the first K elements
            sum_b = sum(pair[1] for pair in pairs[:K])
        else:
            # Update sum_b by removing the previous element and adding the new one
            sum_b = sum_b - pairs[i-1][1] + pairs[i+K-1][1]
        
        # Calculate the result for the current window
        result = pairs[i+K-1][0] * sum_b
        
        # Update min_result if necessary
        min_result = min(min_result, result)
    
    return min_result

# Read the number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    print(solve_case())