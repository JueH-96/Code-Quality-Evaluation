def solve():
    n = int(input())
    s = input()

    dp = [{} for _ in range(n)]

    # Initialize for the first game
    if s[0] == 'R':
        dp[0]['R'] = 0
        dp[0]['P'] = 1
    elif s[0] == 'P':
        dp[0]['P'] = 0
        dp[0]['S'] = 1
    elif s[0] == 'S':
        dp[0]['S'] = 0
        dp[0]['R'] = 1

    for i in range(1, n):
        if s[i] == 'R':
            if 'P' in dp[i-1]:
                dp[i]['R'] = dp[i-1]['P']
            if 'R' in dp[i-1]:
                dp[i]['P'] = dp[i-1]['R'] + 1
        elif s[i] == 'P':
            if 'S' in dp[i-1]:
                dp[i]['P'] = dp[i-1]['S']
            if 'P' in dp[i-1]:
                dp[i]['S'] = dp[i-1]['P'] + 1
        elif s[i] == 'S':
            if 'R' in dp[i-1]:
                dp[i]['S'] = dp[i-1]['R']
            if 'S' in dp[i-1]:
                dp[i]['R'] = dp[i-1]['S'] + 1

    if not dp[n-1]:
        print(0)
    else:
        print(max(dp[n-1].values()))

solve()