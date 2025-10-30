def solve():
    n, d = map(int, input().split())
    s = list(input())
    
    cookies = []
    for i in range(n):
        if s[i] == '@':
            cookies.append(i)
    
    for _ in range(d):
        if cookies:
            last_cookie_index = cookies[-1]
            cookies.pop()
            s[last_cookie_index] = '.'
    
    print("".join(s))

solve()