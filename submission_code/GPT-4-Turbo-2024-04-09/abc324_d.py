import sys
import math
from itertools import permutations

input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

def is_square(x):
    root = int(math.isqrt(x))
    return root * root == x

# Generate all unique permutations of the string S
unique_numbers = set()
for perm in permutations(S):
    num_str = ''.join(perm)
    # Convert to integer and avoid leading zeros issue by using int conversion
    num = int(num_str)
    unique_numbers.add(num)

# Count how many of these numbers are perfect squares
count = 0
for number in unique_numbers:
    if is_square(number):
        count += 1

print(count)