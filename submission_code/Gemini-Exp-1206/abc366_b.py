def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    m = max(len(si) for si in s)
    
    t = ["" for _ in range(m)]
    
    for j in range(m):
        for i in range(n):
            if j < len(s[i]):
                t[j] += s[i][j]
            else:
                t[j] += "*"
    
    for j in range(m):
        t[j] = t[j][::-1]
        while t[j].endswith("*"):
            t[j] = t[j][:-1]
    
    for ti in t:
        print(ti)

solve()