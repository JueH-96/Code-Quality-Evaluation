def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    m = max(len(si) for si in s)
    t = [["*" for _ in range(n)] for _ in range(m)]

    for j in range(m):
        for i in range(n):
            if j < len(s[i]):
                t[j][n - 1 - i] = s[i][j]

    result = []
    for row in t:
        row_str = "".join(row)
        k = len(row_str) - 1
        while k >= 0 and row_str[k] == '*':
            k -= 1
        result.append(row_str[:k+1])

    for row_str in result:
        print(row_str)

solve()