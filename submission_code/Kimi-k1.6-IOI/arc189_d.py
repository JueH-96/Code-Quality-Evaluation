n = int(input())
A = list(map(int, input().split()))

right_sum = [0] * n
j = 0
for i in range(n):
    if i > j:
        current_sum = A[i]
        j = i
    else:
        current_sum = A[i]
    while j + 1 < n and A[j+1] < current_sum:
        current_sum += A[j+1]
        j += 1
    right_sum[i] = current_sum

left_sum = [0] * n
j = n - 1
for i in range(n-1, -1, -1):
    if i < j:
        current_sum = A[i]
        j = i
    else:
        current_sum = A[i]
    while j - 1 >= 0 and A[j-1] < current_sum:
        current_sum += A[j-1]
        j -= 1
    left_sum[i] = current_sum

ans = [left_sum[i] + right_sum[i] - A[i] for i in range(n)]
print(' '.join(map(str, ans)))