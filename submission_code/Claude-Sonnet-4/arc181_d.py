def count_inversions_efficient(arr):
    # Use merge sort approach for O(n log n) inversion counting
    def merge_and_count(arr, temp, left, mid, right):
        i, j, k = left, mid + 1, left
        inv_count = 0
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        
        for i in range(left, right + 1):
            arr[i] = temp[i]
        
        return inv_count
    
    def merge_sort_and_count(arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort_and_count(arr, temp, left, mid)
            inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
            inv_count += merge_and_count(arr, temp, left, mid, right)
        return inv_count
    
    temp = [0] * len(arr)
    arr_copy = arr[:]
    return merge_sort_and_count(arr_copy, temp, 0, len(arr) - 1)

def perform_operation(P, k):
    # Operation k: bubble sort one pass on first k elements
    for i in range(k - 1):
        if P[i] > P[i + 1]:
            P[i], P[i + 1] = P[i + 1], P[i]

# Read input
N = int(input())
P = list(map(int, input().split()))
M = int(input())
A = list(map(int, input().split()))

# Process each operation sequence
for i in range(M):
    # Apply operation A[i]
    perform_operation(P, A[i])
    
    # Count inversions and output
    inversions = count_inversions_efficient(P)
    print(inversions)