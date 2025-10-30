import bisect

n = int(input())
A = list(map(int, input().split()))
sorted_A = sorted(A)
prefix_sum = [0] * n

current_sum = 0
for i in range(n-1, -1, -1):
    current_sum += sorted_A[i]
    prefix_sum[i] = current_sum

result = []
for x in A:
    j = bisect.bisect_right(sorted_A, x)
    if j < n:
        result.append(str(prefix_sum[j]))
    else:
        result.append('0')

print(' '.join(result))