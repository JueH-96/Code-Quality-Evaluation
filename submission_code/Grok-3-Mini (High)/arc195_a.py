import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = [int(data[i]) for i in range(index, index + N)]
index += N
B = [int(data[i]) for i in range(index, index + M)]

dp = [0] * (M + 1)
dp[0] = 1

for num in A:
    for j in range(M, 0, -1):
        if num == B[j - 1]:
            new_val = dp[j] + dp[j - 1]
            if new_val > 2:
                new_val = 2
            dp[j] = new_val

if dp[M] >= 2:
    print("Yes")
else:
    print("No")