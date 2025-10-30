def solve():
    t = input()
    n = int(input())
    bags = []
    for _ in range(n):
        line = input().split()
        bags.append(line[1:])

    dp = {}
    dp[0] = 0

    for i in range(len(t) + 1):
        if i not in dp:
            continue
        
        for j in range(n):
            for s in bags[j]:
                if i + len(s) <= len(t) and t[i:i+len(s)] == s:
                    next_i = i + len(s)
                    if next_i not in dp:
                        dp[next_i] = float('inf')
                    dp[next_i] = min(dp[next_i], dp[i] + 1)

    if len(t) in dp:
        print(dp[len(t)])
    else:
        print(-1)

solve()