def solve():
    s = input()
    n = len(s)
    max_len = 1
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            if sub == sub[::-1]:
                max_len = max(max_len, len(sub))
    print(max_len)

solve()