def solve():
    n, m = map(int, input().split())
    
    families = {}
    
    for i in range(m):
        a, b = input().split()
        a = int(a)
        
        if a not in families:
            families[a] = False
        
        if b == 'M':
            if not families[a]:
                print("Yes")
                families[a] = True
            else:
                print("No")
        else:
            print("No")

solve()