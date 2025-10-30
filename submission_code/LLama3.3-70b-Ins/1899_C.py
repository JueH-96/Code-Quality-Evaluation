import sys

def max_subarray_sum(arr):
    n = len(arr)
    max_sum = float('-inf')
    for i in range(n):
        for j in range(i, n):
            subarray = arr[i:j+1]
            if is_alternating_parity(subarray):
                max_sum = max(max_sum, sum(subarray))
    return max_sum

def is_alternating_parity(arr):
    for i in range(len(arr) - 1):
        if arr[i] % 2 == arr[i+1] % 2:
            return False
    return True

t = int(input())
for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    print(max_subarray_sum(arr))