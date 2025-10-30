# YOUR CODE HERE
from itertools import permutations
import math

def is_square(n):
    return int(math.sqrt(n)) ** 2 == n

N = int(input())
S = input().strip()

unique_squares = set()

for perm in permutations(S):
    if perm[0] == '0':
        continue
    num = int(''.join(perm))
    if is_square(num):
        unique_squares.add(num)

print(len(unique_squares))