import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
M = 100000000
sum_A = sum(A)
sum_part = (N - 1) * sum_A
A_sorted = sorted(A)
count = 0
left = 0
right = N - 1
while left < right:
    current_sum = A_sorted[left] + A_sorted[right]
    if current_sum >= M:
        count += (right - left)
        right -= 1
    else:
        left += 1
result = sum_part - M * count
print(result)