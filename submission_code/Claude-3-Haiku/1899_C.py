import sys

def max_subarray_sum(arr):
    n = len(arr)
    max_sum = float('-inf')
    curr_sum = 0
    prev_parity = None
    
    for i in range(n):
        if prev_parity is None or arr[i] % 2 != prev_parity:
            curr_sum = max(arr[i], curr_sum + arr[i])
            max_sum = max(max_sum, curr_sum)
            prev_parity = arr[i] % 2
        else:
            curr_sum = 0
            prev_parity = None
    
    return max_sum

t = int(input())
for _ in range(t):
    n = int(input())
    arr = [int(x) for x in input().split()]
    print(max_subarray_sum(arr))