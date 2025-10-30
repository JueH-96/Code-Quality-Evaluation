def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    for k in range(n):
        ans = len(s[k])
        for i in range(k):
            for j in range(min(len(s[k]), len(s[i])) + 1):
                if s[k][:j] == s[i][:j]:
                    cost = len(s[k]) - j + len(s[i]) - j
                    ans = min(ans, cost)
        print(ans)

solve()