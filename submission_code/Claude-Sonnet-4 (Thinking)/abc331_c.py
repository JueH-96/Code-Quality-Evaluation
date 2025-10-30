n = int(input())
a = list(map(int, input().split()))

sorted_values = sorted(a)

suffix_sum = [0] * (n + 1)
for i in range(n - 1, -1, -1):
    suffix_sum[i] = suffix_sum[i + 1] + sorted_values[i]

def find_first_greater(arr, x):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= x:
            left = mid + 1
        else:
            right = mid
    return left

result = []
for val in a:
    pos = find_first_greater(sorted_values, val)
    result.append(suffix_sum[pos])

print(' '.join(map(str, result)))