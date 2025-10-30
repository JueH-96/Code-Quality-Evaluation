def solve():
    n, d = map(int, input().split())
    s = input()
    
    cookie_count = s.count('@')
    empty_count = n - cookie_count
    
    empty_count += min(d, cookie_count)
    
    print(empty_count)

solve()