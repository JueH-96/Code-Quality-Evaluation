# YOUR CODE HERE
def can_make_non_decreasing(n, arr):
    total_sum = sum(arr)
    
    # Try to construct the lexicographically smallest non-decreasing sequence
    # with the same sum
    target = [0] * n
    remaining_sum = total_sum
    
    for i in range(n):
        remaining_positions = n - i
        # Each element should be at least the ceiling of remaining_sum / remaining_positions
        min_val = (remaining_sum + remaining_positions - 1) // remaining_positions
        
        if i > 0:
            # Also need to be at least the previous element
            min_val = max(min_val, target[i-1])
        
        target[i] = min_val
        remaining_sum -= min_val
        
        if remaining_sum < 0:
            return False
    
    # Now check if we can transform arr to target
    # We need to check if at each prefix, we have enough sum
    prefix_sum_arr = 0
    prefix_sum_target = 0
    
    for i in range(n):
        prefix_sum_arr += arr[i]
        prefix_sum_target += target[i]
        
        # We need at least this much sum in the prefix to achieve the target
        if prefix_sum_arr < prefix_sum_target:
            return False
    
    return True

T = int(input())
for _ in range(T):
    N = int(input())
    A = list(map(int, input().split()))
    
    if can_make_non_decreasing(N, A):
        print("Yes")
    else:
        print("No")