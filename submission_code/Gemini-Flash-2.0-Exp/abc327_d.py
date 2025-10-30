def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for i in range(2**n):
        x = []
        for j in range(n):
            if (i >> j) & 1:
                x.append(1)
            else:
                x.append(0)
        
        valid = True
        for k in range(m):
            if x[a[k]-1] == x[b[k]-1]:
                valid = False
                break
        
        if valid:
            print("Yes")
            return
    
    print("No")

solve()