import sys

def max_subarray_sum(arr, left, right):
    max_ending_here = arr[left]
    max_so_far = arr[left]
    for i in range(left + 1, right + 1):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
    return max_so_far

def max_alternating_sum(arr):
    n = len(arr)
    max_sum_ans = float('-inf')
    start_idx = 0
    for i in range(n - 1):
        if (arr[i] % 2) == (arr[i + 1] % 2):  # same parity
            seg_max = max_subarray_sum(arr, start_idx, i)
            if seg_max > max_sum_ans:
                max_sum_ans = seg_max
            start_idx = i + 1
    # Add the last segment
    seg_max = max_subarray_sum(arr, start_idx, n - 1)
    if seg_max > max_sum_ans:
        max_sum_ans = seg_max
    return max_sum_ans

# Read input
data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1
for _ in range(t):
    n = int(data[index])
    index += 1
    arr = list(map(int, data[index:index + n]))
    index += n
    # Compute and print the answer
    ans = max_alternating_sum(arr)
    print(ans)