# YOUR CODE HERE
import itertools
import math

def solve():
    n = int(input())
    s = input()
    
    digits = [int(d) for d in s]
    
    unique_perms = set()
    count = 0
    
    for perm in itertools.permutations(digits):
        num_str = "".join(map(str, perm))
        num = int(num_str)
        
        if num_str not in unique_perms:
            unique_perms.add(num_str)
            sqrt_num = int(math.sqrt(num))
            if sqrt_num * sqrt_num == num:
                count += 1
    
    print(count)

solve()