def solve():
    n, m = map(int, input().split())
    families = [False] * n
    
    for _ in range(m):
        a, b = input().split()
        a = int(a) - 1
        
        if b == 'M':
            if not families[a]:
                print("Yes")
                families[a] = True
            else:
                print("No")
        else:
            print("No")

solve()