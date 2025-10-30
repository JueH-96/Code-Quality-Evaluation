def solve():
    n = int(input())
    s = input().split()
    
    ans = 0
    for i in range(n):
        for j in range(i + 1, n):
            lcp = 0
            for k in range(min(len(s[i]), len(s[j]))):
                if s[i][k] == s[j][k]:
                    lcp += 1
                else:
                    break
            ans += lcp
    print(ans)

solve()