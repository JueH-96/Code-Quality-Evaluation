import itertools
import math

def solve():
    n = int(input())
    s = input()
    digits = [int(c) for c in s]
    
    unique_squares = set()
    
    for perm in itertools.permutations(digits):
        num = 0
        for i in range(n):
            num = num * 10 + perm[i]
        
        sqrt_num = math.isqrt(num)
        if sqrt_num * sqrt_num == num:
            unique_squares.add(num)
            
    print(len(unique_squares))

solve()