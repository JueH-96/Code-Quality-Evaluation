import sys
from collections import defaultdict

def read_ints(): return list(map(int, sys.stdin.readline().strip().split()))

def count_swaps_to_sort(arr):
    sorted_arr = sorted(arr)
    swaps = 0
    index_dict = {v: i for i, v in enumerate(arr)}
    for i in range(len(arr)):
        if arr[i] != sorted_arr[i]:
            swaps += 1
            # Swap current element with correct element
            to_swap_idx = index_dict[sorted_arr[i]]
            arr[i], arr[to_swap_idx] = arr[to_swap_idx], arr[i]
            # Update the indices in the dictionary
            index_dict[arr[to_swap_idx]] = to_swap_idx
            index_dict[arr[i]] = i
    return swaps

def min_swaps_to_match_grids(A, B, H, W):
    # Transpose B to make it easier to compare with A
    B_transposed = list(zip(*B))
    
    # Count swaps for rows
    row_swaps = 0
    for row in A:
        row_swaps += count_swaps_to_sort(row)
    
    # Count swaps for columns
    col_swaps = 0
    for col in B_transposed:
        col_swaps += count_swaps_to_sort(col)
    
    # Check if the sorted versions of A and B are equal
    A_sorted = [sorted(row) for row in A]
    B_sorted = [sorted(col) for col in B_transposed]
    if A_sorted != B_sorted:
        return -1
    
    return row_swaps + col_swaps

H, W = read_ints()
A = [read_ints() for _ in range(H)]
B = [read_ints() for _ in range(H)]

print(min_swaps_to_match_grids(A, B, H, W))