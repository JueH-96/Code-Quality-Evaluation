def solve():
    n, d = map(int, input().split())
    s = list(input())

    for _ in range(d):
        rightmost_cookie_index = -1
        for i in range(n - 1, -1, -1):
            if s[i] == '@':
                rightmost_cookie_index = i
                break
        if rightmost_cookie_index != -1:
            s[rightmost_cookie_index] = '.'

    print("".join(s))

solve()