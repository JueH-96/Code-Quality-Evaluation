def operation_k(perm, k):
    for i in range(k-1):  # This corresponds to i from 1 to k-1 in 1-indexed
        if perm[i] > perm[i+1]:
            perm[i], perm[i+1] = perm[i+1], perm[i]
    return perm

def merge_sort_and_count(arr):
    if len(arr) <= 1:
        return arr, 0
    
    mid = len(arr) // 2
    left, inv_left = merge_sort_and_count(arr[:mid])
    right, inv_right = merge_sort_and_count(arr[mid:])
    merged, inv_merge = merge_and_count(left, right)
    
    return merged, inv_left + inv_right + inv_merge

def merge_and_count(left, right):
    merged = []
    inv_count = 0
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv_count += len(left) - i
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged, inv_count

def inversion_number_optimized(perm):
    _, inv_count = merge_sort_and_count(perm.copy())
    return inv_count

def solve(n, p, m, a):
    results = []
    current_p = p.copy()  # Make a copy at the beginning
    for op in a:
        operation_k(current_p, op)  # Modify current_p in-place
        results.append(inversion_number_optimized(current_p))
    return results

n = int(input())
p = list(map(int, input().split()))
m = int(input())
a = list(map(int, input().split()))

results = solve(n, p, m, a)
for res in results:
    print(res)