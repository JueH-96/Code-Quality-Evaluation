# YOUR CODE HERE
T = input()
N = int(input())
bags = []
for i in range(N):
    line = input().split()
    A = int(line[0])
    strings = line[1:]
    bags.append(strings)

dp = {}
dp[""] = 0

for i in range(N):
    new_dp = {}
    for s, cost in dp.items():
        new_dp[s] = cost
        for string in bags[i]:
            new_s = s + string
            if new_s in new_dp:
                new_dp[new_s] = min(new_dp[new_s], cost + 1)
            else:
                new_dp[new_s] = cost + 1
    dp = new_dp

if T in dp:
    print(dp[T])
else:
    print(-1)