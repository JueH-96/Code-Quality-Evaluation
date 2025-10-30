def solve():
    n = int(input())
    
    s = ""
    while n > 0:
        s = str((n % 5) * 2) + s
        n = (n - 1) // 5
    
    print(s)

solve()