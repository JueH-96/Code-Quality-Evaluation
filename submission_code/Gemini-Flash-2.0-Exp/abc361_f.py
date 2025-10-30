def solve():
    n = int(input())
    
    count = 0
    seen = set()
    
    for a in range(2, int(n**(1/2)) + 2):
        b = 2
        val = a**b
        while val <= n:
            if val not in seen:
                count += 1
                seen.add(val)
            b += 1
            if n // a < val // a:
                break
            val = a**b
            
    print(count)

solve()