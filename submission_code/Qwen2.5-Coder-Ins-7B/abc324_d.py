import sys
from itertools import permutations

# Read input
N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

# Generate all permutations of the string S
perms = permutations(S)

# Function to check if a number is a perfect square
def is_perfect_square(n):
    root = int(n**0.5)
    return root * root == n

# Count the number of permutations that form a perfect square
count = 0
for perm in perms:
    num = int(''.join(perm))
    if is_perfect_square(num):
        count += 1

# Output the result
print(count)