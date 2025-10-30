from itertools import permutations
import math

def is_perfect_square(n):
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

n = int(input())
s = input().strip()

digits = [int(c) for c in s]
square_numbers = set()

for perm in permutations(digits):
    num = 0
    for digit in perm:
        num = num * 10 + digit
    
    if is_perfect_square(num):
        square_numbers.add(num)

print(len(square_numbers))