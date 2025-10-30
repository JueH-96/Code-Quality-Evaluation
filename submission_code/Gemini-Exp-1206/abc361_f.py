def solve():
    n = int(input())
    
    s = set()
    
    for a in range(1, 1000001):
        for b in range(2, 61):
            x = a ** b
            if x > n:
                break
            s.add(x)
            
    print(len(s))

solve()