def solve():
    n, a, b = map(int, input().split())
    d = list(map(int, input().split()))
    
    week_len = a + b
    
    for day in d:
        if (day % week_len) == 0:
            if a < week_len:
                print("No")
                return
        elif (day % week_len) > a:
            print("No")
            return
    
    print("Yes")

solve()