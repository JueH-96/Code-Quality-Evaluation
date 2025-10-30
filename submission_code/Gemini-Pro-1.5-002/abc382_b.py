def solve():
    n, d = map(int, input().split())
    s = list(input())
    
    cookies_eaten = 0
    while cookies_eaten < d:
        for i in range(n - 1, -1, -1):
            if s[i] == '@':
                s[i] = '.'
                cookies_eaten += 1
                break
    
    print("".join(s))

solve()