def min_operations_to_equalize(arr, n):
    # If there's only one element, the difference is already 0
    if n == 1:
        return 0
    
    # Calculate the target values
    total_sum = sum(arr)
    avg_floor = total_sum // n
    avg_ceil = avg_floor + 1
    
    # Determine how many elements should be avg_floor and how many should be avg_ceil
    # After operations, the sum remains the same
    num_ceil = total_sum % n
    num_floor = n - num_ceil
    
    # Sort the array to efficiently distribute values
    arr.sort()
    
    # Calculate the total amount we need to increase elements
    # (which is equivalent to the amount we need to decrease other elements)
    total_increases = 0
    
    for i in range(n):
        if i < num_floor:
            # This element should become avg_floor
            if arr[i] < avg_floor:
                total_increases += avg_floor - arr[i]
        else:
            # This element should become avg_ceil
            if arr[i] < avg_ceil:
                total_increases += avg_ceil - arr[i]
    
    return total_increases

# Read input
n = int(input())
arr = list(map(int, input().split()))

# Calculate and print the answer
print(min_operations_to_equalize(arr, n))