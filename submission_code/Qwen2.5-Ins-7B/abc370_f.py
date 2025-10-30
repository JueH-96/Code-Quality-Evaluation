# N K
# A_1 A_2 ... A_N
N, K = map(int, input().split())
A = list(map(int, input().split()))

total_mass = sum(A)
A_sum = [0] * (N + 1)
for i in range(N):
    A_sum[i + 1] = A_sum[i] + A[i]

def check(x):
    dp = [0] * (N + 1)
    for i in range(N):
        dp[i + 1] = dp[i] + A[i]
        if dp[i + 1] > x:
            return False
        if dp[i + 1] == x:
            dp[i + 1] = 0
    return True

low, high = total_mass // K, total_mass
while low < high:
    mid = (low + high + 1) // 2
    if check(mid):
        low = mid
    else:
        high = mid - 1

ans = low
cut_lines = 0
for i in range(N):
    if A_sum[i + 1] > ans and A_sum[i] <= ans:
        cut_lines += 1

print(ans, N - cut_lines)