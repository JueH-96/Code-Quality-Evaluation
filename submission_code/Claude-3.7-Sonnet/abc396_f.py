def count_inversions(arr):
    """Count inversions in an array using merge sort."""
    arr_copy = arr.copy()
    return merge_sort_and_count(arr_copy, 0, len(arr_copy) - 1)

def merge_sort_and_count(arr, left, right):
    """Merge sort the array and count inversions."""
    if left >= right:
        return 0
    
    mid = (left + right) // 2
    count = merge_sort_and_count(arr, left, mid)
    count += merge_sort_and_count(arr, mid + 1, right)
    count += merge_and_count(arr, left, mid, right)
    return count

def merge_and_count(arr, left, mid, right):
    """Merge two sorted arrays and count inversions."""
    # Create temporary arrays for left and right parts
    L = arr[left:mid+1]
    R = arr[mid+1:right+1]
    
    # Indices for the temporary arrays and the merged array
    i = j = 0
    k = left
    inv_count = 0
    
    # Merge the temporary arrays back
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
            inv_count += len(L) - i
        k += 1
    
    # Copy the remaining elements of L, if any
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    
    # Copy the remaining elements of R, if any
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1
    
    return inv_count

# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Compute the inversion number for each k
for k in range(M):
    B = [(a + k) % M for a in A]
    inversions = count_inversions(B)
    print(inversions)