import sys
import bisect

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
A = [int(data[i]) for i in range(index, index + N)]

# Compute sum of all A_i
sum_A = sum(A)

# If sum of A_i <= M, subsidy limit can be infinite
if sum_A <= M:
    print("infinite")
else:
    # Sort A in ascending order
    A_sorted = sorted(A)
    
    # Compute prefix sum
    prefix_sum = [0]
    cumsum = 0
    for num in A_sorted:
        cumsum += num
        prefix_sum.append(cumsum)
    
    # Maximum value in A
    max_A = A_sorted[-1]
    
    # Binary search to find maximum x
    left = 0
    right = max_A
    ans = 0  # Initialize answer to 0, as sum_min(0) = 0 <= M always
    
    while left <= right:
        mid = (left + right) // 2
        # Find number of elements <= mid using bisect
        k = bisect.bisect_right(A_sorted, mid)
        sum_leq = prefix_sum[k]  # Sum of elements <= mid
        num_greater = N - k  # Number of elements > mid
        sum_min_x = sum_leq + mid * num_greater  # Sum of min(mid, A_i)
        
        if sum_min_x <= M:
            ans = mid
            left = mid + 1  # Try larger x
        else:
            right = mid - 1  # Try smaller x
    
    # Output the answer
    print(ans)