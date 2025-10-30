# YOUR CODE HERE
def count_inversions(arr):
    # Function to count inversions using merge sort
    if len(arr) < 2:
        return arr, 0
    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])
    merged, split_inv = merge_and_count(left, right)
    return merged, left_inv + right_inv + split_inv

def merge_and_count(left, right):
    merged = []
    i = j = 0
    inversions = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inversions += len(left) - i
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inversions

def apply_operation_k(P, k):
    for i in range(k - 1):
        if P[i] > P[i + 1]:
            P[i], P[i + 1] = P[i + 1], P[i]

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:N+1]))
M = int(data[N+1])
A = list(map(int, data[N+2:N+2+M]))

for a in A:
    apply_operation_k(P, a)
    _, inversions = count_inversions(P)
    print(inversions)