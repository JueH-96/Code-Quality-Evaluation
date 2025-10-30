# YOUR CODE HERE
import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index+1])
index += 2
S = data[index]
T = data[index+1]

DP = [False] * (N + 1)
DP[0] = True

for i in range(1, N+1):
    for l in range(1, M+1):
        if i - l < 0:
            continue
        for c in range(M):
            if c + l > M:
                continue
            if S[i-l:i] == T[c:c+l]:
                s_start = (i - l) - c
                if s_start >= 0 and DP[s_start]:
                    DP[i] = True
                    break
                elif s_start < 0 and i - l == 0 and c == 0:
                    DP[i] = True
                    break
        if DP[i]:
            break  # Break after setting DP[i] to True to optimize

if DP[N]:
    print("Yes")
else:
    print("No")
# END OF CODE