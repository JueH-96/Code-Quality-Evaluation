def is_400_number(n):
    if n <= 0:
        return False
    
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    if len(factors) != 2:
        return False
    
    for count in factors.values():
        if count % 2 != 0:
            return False
    
    return True

def solve():
    q = int(input())
    for _ in range(q):
        a = int(input())
        
        while not is_400_number(a):
            a -= 1
        
        print(a)

solve()