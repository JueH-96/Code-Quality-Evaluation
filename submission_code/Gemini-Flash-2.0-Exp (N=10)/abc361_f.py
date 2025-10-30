def solve():
    n = int(input())
    
    s = set()
    
    for a in range(2, int(n**0.5) + 2):
        curr = a * a
        while curr <= n:
            s.add(curr)
            if curr > n // a:
                break
            curr *= a
    
    print(len(s))

solve()