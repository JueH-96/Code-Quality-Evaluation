import math

def solve():
    n = int(input())
    count = 0
    
    max_b = int(math.sqrt(n))
    
    seen = set()
    
    for a in range(1, 61):
        two_pow_a = 2**a
        if two_pow_a > n:
            break
        
        max_b = int(math.sqrt(n / two_pow_a))
        
        for b in range(1, max_b + 1):
            x = two_pow_a * (b**2)
            if 1 <= x <= n:
                if x not in seen:
                    count += 1
                    seen.add(x)
    print(count)

solve()