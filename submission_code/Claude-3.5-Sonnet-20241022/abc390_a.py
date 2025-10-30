# Read input
A = list(map(int, input().split()))

def can_sort_with_one_swap(arr):
    # If already sorted, we can't perform a swap
    if arr == [1, 2, 3, 4, 5]:
        return False
    
    # Try swapping each adjacent pair
    for i in range(len(arr)-1):
        # Create a copy of array
        test_arr = arr.copy()
        # Swap adjacent elements
        test_arr[i], test_arr[i+1] = test_arr[i+1], test_arr[i]
        # Check if array is sorted after swap
        if test_arr == [1, 2, 3, 4, 5]:
            return True
    
    return False

# Print result
if can_sort_with_one_swap(A):
    print("Yes")
else:
    print("No")