def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    total_sum = sum(A)
    
    # Compute prefix sums of A
    prefix_A = [0] * (N + 1)
    for i in range(N):
        prefix_A[i + 1] = prefix_A[i] + A[i]
    
    # Try a uniform distribution of the total sum
    avg = total_sum // N
    remainder = total_sum % N
    
    # Create target non-decreasing sequence
    B = [avg] * N
    # Add remainder to rightmost positions to maintain non-decreasing property
    for i in range(remainder):
        B[N - 1 - i] += 1
    
    # Check if this B satisfies the prefix constraints
    # For each prefix, sum(B[1..i]) should be achievable from sum(A[1..i])
    prefix_B = 0
    for i in range(N):
        prefix_B += B[i]
        # We need prefix_B >= prefix_A[i+1] to be feasible
        # (we can only increase prefix sums by moving value from right to left)
        if i < N - 1 and prefix_B < prefix_A[i + 1]:
            return "No"
    
    return "Yes"

T = int(input())
for _ in range(T):
    print(solve())