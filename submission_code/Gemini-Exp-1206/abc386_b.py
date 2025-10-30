def solve():
    s = input()
    n = len(s)
    ans = n
    i = 0
    while i < n:
        if i + 1 < n and s[i:i+2] == "00":
            ans -= 1
            i += 2
        else:
            i += 1
    print(ans)

solve()