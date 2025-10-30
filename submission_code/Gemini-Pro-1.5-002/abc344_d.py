# YOUR CODE HERE
def solve():
    t = input()
    n = int(input())
    bags = []
    for _ in range(n):
        line = input().split()
        bags.append(line[1:])

    dp = [[float('inf')] * (len(t) + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(n):
        for j in range(len(t) + 1):
            if dp[i][j] == float('inf'):
                continue
            
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
            
            for s in bags[i]:
                if j + len(s) <= len(t) and t[j:j+len(s)] == s:
                    dp[i+1][j+len(s)] = min(dp[i+1][j+len(s)], dp[i][j] + 1)

    if dp[n][len(t)] == float('inf'):
        print(-1)
    else:
        print(dp[n][len(t)])

solve()