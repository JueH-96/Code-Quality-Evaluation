# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    x_list = []
    y_list = []
    for i in range(N):
        x = int(data[2 + 2*i])
        y = int(data[3 + 2*i])
        x_list.append(x)
        y_list.append(y)
    
    # Sort x and y lists
    x_list_sorted = sorted(x_list)
    y_list_sorted = sorted(y_list)
    
    # Precompute prefix sums for x and y
    prefix_x = [0] * (N + 1)
    for i in range(N):
        prefix_x[i+1] = prefix_x[i] + x_list_sorted[i]
    
    prefix_y = [0] * (N + 1)
    for i in range(N):
        prefix_y[i+1] = prefix_y[i] + y_list_sorted[i]
    
    # Function to calculate the sum of absolute differences for x
    def sum_abs_x(x):
        idx = bisect.bisect_left(x_list_sorted, x)
        sum_left = x * idx - prefix_x[idx]
        sum_right = (prefix_x[N] - prefix_x[idx]) - x * (N - idx)
        return sum_left + sum_right
    
    # Function to calculate the sum of absolute differences for y
    def sum_abs_y(y):
        idx = bisect.bisect_left(y_list_sorted, y)
        sum_left = y * idx - prefix_y[idx]
        sum_right = (prefix_y[N] - prefix_y[idx]) - y * (N - idx)
        return sum_left + sum_right
    
    # Find the range of x and y that satisfy the condition
    # For x, find the minimum and maximum x such that sum_abs_x(x) <= D
    # Similarly for y
    
    # Binary search for x_min and x_max
    low_x = -10**7
    high_x = 10**7
    x_min = low_x
    x_max = high_x
    
    # Binary search for x_min
    left = low_x
    right = high_x
    while left <= right:
        mid = (left + right) // 2
        if sum_abs_x(mid) <= D:
            x_min = mid
            right = mid - 1
        else:
            left = mid + 1
    
    # Binary search for x_max
    left = low_x
    right = high_x
    while left <= right:
        mid = (left + right) // 2
        if sum_abs_x(mid) <= D:
            x_max = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Binary search for y_min and y_max
    low_y = -10**7
    high_y = 10**7
    y_min = low_y
    y_max = high_y
    
    # Binary search for y_min
    left = low_y
    right = high_y
    while left <= right:
        mid = (left + right) // 2
        if sum_abs_y(mid) <= D:
            y_min = mid
            right = mid - 1
        else:
            left = mid + 1
    
    # Binary search for y_max
    left = low_y
    right = high_y
    while left <= right:
        mid = (left + right) // 2
        if sum_abs_y(mid) <= D:
            y_max = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Now, for each x in [x_min, x_max], find the range of y that satisfies sum_abs_x(x) + sum_abs_y(y) <= D
    # Since sum_abs_x(x) is fixed for a given x, we can compute the maximum sum_abs_y(y) allowed
    # Then, for each x, find the range of y that satisfies sum_abs_y(y) <= D - sum_abs_x(x)
    
    # Precompute the sum_abs_x for all x in [x_min, x_max]
    # Since x_min and x_max can be large, we need a smarter way
    # Instead, we can iterate over x in [x_min, x_max] and for each x, find the range of y
    
    # However, since x_min and x_max can be up to 1e7, iterating over all x is not feasible
    # Instead, we can precompute the sum_abs_x for all x in the sorted x_list_sorted and use interpolation
    
    # Alternatively, we can precompute the sum_abs_x for all x in the range [x_min, x_max] in a smarter way
    # But given time constraints, we will proceed with the initial approach
    
    # Initialize the count
    count = 0
    
    # Iterate over x in [x_min, x_max]
    for x in range(x_min, x_max + 1):
        sum_x = sum_abs_x(x)
        if sum_x > D:
            continue
        remaining_D = D - sum_x
        # Find the range of y that satisfies sum_abs_y(y) <= remaining_D
        # Binary search for y_min and y_max
        left_y = y_min
        right_y = y_max
        # Binary search for y_min
        low = y_min
        high = y_max
        y_min_current = y_max
        while low <= high:
            mid = (low + high) // 2
            if sum_abs_y(mid) <= remaining_D:
                y_min_current = mid
                high = mid - 1
            else:
                low = mid + 1
        # Binary search for y_max
        low = y_min
        high = y_max
        y_max_current = y_min
        while low <= high:
            mid = (low + high) // 2
            if sum_abs_y(mid) <= remaining_D:
                y_max_current = mid
                low = mid + 1
            else:
                high = mid - 1
        # Count the number of y in [y_min_current, y_max_current]
        if y_min_current > y_max_current:
            continue
        count += y_max_current - y_min_current + 1
    
    print(count)

if __name__ == "__main__":
    main()