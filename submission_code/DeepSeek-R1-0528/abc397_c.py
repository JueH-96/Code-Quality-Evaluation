n = int(input().strip())
A = list(map(int, input().split()))

left = [0] * (n + 1)
freq_left = [0] * (n + 1)
distinct_left = 0
for i in range(n):
    num = A[i]
    freq_left[num] += 1
    if freq_left[num] == 1:
        distinct_left += 1
    left[i + 1] = distinct_left

right_arr = [0] * (n + 1)
freq_right = [0] * (n + 1)
distinct_right = 0
for i in range(n - 1, -1, -1):
    num = A[i]
    freq_right[num] += 1
    if freq_right[num] == 1:
        distinct_right += 1
    right_arr[i] = distinct_right

ans = 0
for i in range(1, n):
    total = left[i] + right_arr[i]
    if total > ans:
        ans = total

print(ans)