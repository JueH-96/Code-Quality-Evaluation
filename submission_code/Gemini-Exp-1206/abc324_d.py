import itertools

def is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    x = int(n**0.5)
    return x * x == n

def solve():
    n = int(input())
    s = input()
    
    count = 0
    seen = set()
    
    for p in itertools.permutations(s):
        num_str = "".join(p)
        if num_str[0] == '0' and len(num_str) > 1:
            continue
        
        num = int(num_str)
        
        if num not in seen and is_square(num):
            count += 1
            seen.add(num)
            
    print(count)

solve()