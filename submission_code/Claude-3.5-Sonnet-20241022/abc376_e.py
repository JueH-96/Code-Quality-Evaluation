def solve_test_case():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Create pairs of (A[i], B[i]) and sort by A[i]
    pairs = sorted(zip(A, B))
    
    min_result = float('inf')
    
    # For each possible maximum value of A (trying each A[i] as potential max)
    for i in range(N - K + 1):
        max_a = pairs[i + K - 1][0]  # This will be our maximum A value
        
        # Get K elements with smallest B values among elements with A <= max_a
        available_b = [p[1] for p in pairs[:i + K]]
        available_b.sort()  # Sort B values
        b_sum = sum(available_b[:K])  # Take K smallest B values
        
        min_result = min(min_result, max_a * b_sum)
    
    return min_result

# Read number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    result = solve_test_case()
    print(result)