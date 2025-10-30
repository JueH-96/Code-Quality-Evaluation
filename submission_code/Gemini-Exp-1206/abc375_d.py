def solve():
    s = input()
    n = len(s)
    count = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if s[i] == s[k]:
                    count += 1
    print(count)

solve()