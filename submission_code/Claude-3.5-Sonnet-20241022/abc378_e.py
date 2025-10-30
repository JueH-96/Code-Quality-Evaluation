def solve():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Calculate prefix sums modulo M
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % M
    
    # Calculate sum of all subarray sums modulo M
    result = 0
    for l in range(N):
        for r in range(l, N):
            # For subarray A[l:r+1], we can use prefix_sum to get sum modulo M
            # The sum of elements from l to r is:
            # (prefix_sum[r+1] - prefix_sum[l] + M) % M
            subarray_sum = (prefix_sum[r + 1] - prefix_sum[l] + M) % M
            result += subarray_sum
    
    print(result)

# Run the solution
solve()