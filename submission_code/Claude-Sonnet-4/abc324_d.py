import itertools
import math

n = int(input())
s = input().strip()

perfect_squares = set()

# Generate all permutations of the string
for perm in itertools.permutations(s):
    # Convert permutation to integer
    num_str = ''.join(perm)
    num = int(num_str)
    
    # Check if it's a perfect square
    sqrt_num = int(math.sqrt(num))
    if sqrt_num * sqrt_num == num:
        perfect_squares.add(num)

print(len(perfect_squares))