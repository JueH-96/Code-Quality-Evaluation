def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    m = max(len(x) for x in s)
    
    t = ["" for _ in range(m)]
    
    for j in range(m):
        for i in range(n):
            if j < len(s[i]):
                t[j] += s[i][j]
            else:
                t[j] += "*"
    
    for j in range(m):
        t[j] = t[j][::-1]
        while len(t[j]) > 0 and t[j][-1] == '*':
            t[j] = t[j][:-1]
        print(t[j])

solve()