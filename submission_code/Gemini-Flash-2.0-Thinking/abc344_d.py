def solve():
    T = input()
    N = int(input())
    bags = []
    for _ in range(N):
        line = input().split()
        bags.append(line[1:])

    m = len(T)
    dp = [[float('inf')] * (m + 1) for _ in range(N + 1)]
    dp[0][0] = 0

    for i in range(1, N + 1):
        for j in range(m + 1):
            # Do nothing
            dp[i][j] = min(dp[i][j], dp[i - 1][j])

            # Select a string from bag i
            for s in bags[i - 1]:
                ls = len(s)
                if j >= ls and T[j - ls:j] == s and dp[i - 1][j - ls] != float('inf'):
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - ls] + 1)

    if dp[N][m] == float('inf'):
        print(-1)
    else:
        print(dp[N][m])

# YOUR CODE HERE
solve()