def can_sort_by_one_swap(A):
    # Check how many pairs are out of order
    inversions = 0
    swap_index = -1
    for i in range(4):
        if A[i] > A[i + 1]:
            inversions += 1
            swap_index = i
    
    # If there's exactly one inversion, check if swapping the adjacent elements sorts the array
    if inversions == 1:
        # Perform the swap
        A[swap_index], A[swap_index + 1] = A[swap_index + 1], A[swap_index]
        # Check if the array is sorted
        if all(A[i] < A[i + 1] for i in range(4)):
            return "Yes"
        else:
            return "No"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
data = input().strip()
A = list(map(int, data.split()))

# Output the result
print(can_sort_by_one_swap(A))