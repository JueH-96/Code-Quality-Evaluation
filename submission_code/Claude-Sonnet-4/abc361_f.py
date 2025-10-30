import math

def solve(n):
    if n == 1:
        return 1
    
    perfect_powers = set()
    
    # For each exponent b >= 2
    b = 2
    while True:
        # Find the maximum base a such that a^b <= n
        max_a = int(n ** (1.0 / b))
        
        if max_a < 2:
            break
            
        # Add all perfect b-th powers to the set
        for a in range(2, max_a + 1):
            power = a ** b
            if power <= n:
                perfect_powers.add(power)
            else:
                break
        
        b += 1
    
    # Add 1 (which is 1^b for any b >= 2)
    perfect_powers.add(1)
    
    return len(perfect_powers)

n = int(input())
print(solve(n))