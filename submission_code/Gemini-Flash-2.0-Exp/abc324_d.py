import itertools

def solve():
    n = int(input())
    s = input()
    
    digits = sorted(list(s))
    
    count = 0
    seen = set()
    
    for perm in itertools.permutations(digits):
        num_str = "".join(perm)
        
        # Skip leading zeros
        if num_str[0] == '0' and len(num_str) > 1:
            continue
        
        num = int(num_str)
        
        if num not in seen:
            sqrt_num = int(num**0.5)
            if sqrt_num * sqrt_num == num:
                count += 1
                seen.add(num)
    
    print(count)

solve()