def solve():
    n, d = map(int, input().split())
    s = input()
    
    cookies = s.count('@')
    empty_boxes = n - cookies
    
    print(empty_boxes + d)

solve()