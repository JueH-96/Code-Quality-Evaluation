def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    for k in range(n):
        ans = len(s[k])
        for i in range(k):
            lcs = 0
            for j in range(min(len(s[k]), len(s[i]))):
                if s[k][len(s[k]) - 1 - j] == s[i][len(s[i]) - 1 - j]:
                    lcs += 1
                else:
                    break
            ans = min(ans, len(s[k]) - lcs + len(s[i]) - lcs)
        print(ans)

solve()