import itertools

def solve():
    n = int(input())
    s = input()
    digits = [int(d) for d in s]
    
    count = 0
    seen = set()
    
    for p in itertools.permutations(digits):
        num = 0
        for i in range(n):
            num = num * 10 + p[i]
        
        if num == 0:
            continue

        sqrt_num = int(num**0.5)
        if sqrt_num * sqrt_num == num:
            if num not in seen:
                count += 1
                seen.add(num)

    print(count)

solve()