def count_inversions(arr):
    # Function to use merge sort and count inversions
    if len(arr) < 2:
        return arr, 0
    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])
    merged, split_inv = merge_and_count(left, right)
    return merged, left_inv + right_inv + split_inv

def merge_and_count(left, right):
    i = j = inv_count = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inv_count += len(left) - i  # Count inversions
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inv_count

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2 + N]))
    
    results = []
    
    for k in range(M):
        B = [(A[i] + k) % M for i in range(N)]
        _, inversions = count_inversions(B)
        results.append(inversions)
    
    print('
'.join(map(str, results)))