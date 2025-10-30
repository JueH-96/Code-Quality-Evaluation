def solve():
    N, S = map(int, input().split())
    A = list(map(int, input().split()))
    
    period_sum = sum(A)
    
    # Create a repeated version of the array to handle subsequences that wrap around
    repeated_A = A * 2
    
    # Calculate prefix sums for efficient range sum calculation
    prefix_sum = [0]
    for num in repeated_A:
        prefix_sum.append(prefix_sum[-1] + num)
    
    # Check all possible subsequences that start in the first period
    for i in range(N):
        for r in range(2*N):
            if i + r >= len(prefix_sum) - 1:
                break
            
            sub_sum = prefix_sum[i + r + 1] - prefix_sum[i]
            
            # Check if the subsequence itself sums to S
            if sub_sum == S:
                return "Yes"
            
            # Check if adding or subtracting complete periods can get the sum to S
            if period_sum != 0:
                remaining = S - sub_sum
                
                if remaining % period_sum == 0:
                    k = remaining // period_sum
                    if k > 0:
                        if (period_sum > 0 and remaining > 0) or (period_sum < 0 and remaining < 0):
                            return "Yes"
    
    return "No"

print(solve())