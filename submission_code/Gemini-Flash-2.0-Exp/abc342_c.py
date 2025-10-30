def solve():
    n = int(input())
    s = input()
    q = int(input())
    
    for _ in range(q):
        c, d = input().split()
        s = s.replace(c, d)
    
    print(s)

solve()