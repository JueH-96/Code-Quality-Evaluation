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

def apply_operation(P, k):
    for i in range(k - 1):
        if P[i] > P[i + 1]:
            P[i], P[i + 1] = P[i + 1], P[i]
    return P

import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    P = list(map(int, data[index:index + N]))
    index += N
    M = int(data[index])
    index += 1
    A = list(map(int, data[index:index + M]))
    
    results = []
    for i in range(M):
        k = A[i]
        P = apply_operation(P, k)
        _, inversions = count_inversions(P)
        results.append(inversions)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()