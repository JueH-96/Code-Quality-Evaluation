import math
from itertools import permutations

def is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    root = int(math.sqrt(n))
    return root * root == n

def solve():
    n = int(input())
    s = input()
    digits_str = list(s)
    digits = [int(d) for d in digits_str]
    square_number_set = set()
    generated_numbers = set()
    
    for p in permutations(digits):
        num_str_list = [str(d) for d in p]
        num_str = "".join(num_str_list)
        if len(num_str) > 1 and num_str[0] == '0':
            num_str = str(int(num_str))
            
        num = int(num_str)
        if num in generated_numbers:
            continue
        generated_numbers.add(num)
        if is_square(num):
            square_number_set.add(num)
            
    print(len(square_number_set))

if __name__ == '__main__':
    solve()