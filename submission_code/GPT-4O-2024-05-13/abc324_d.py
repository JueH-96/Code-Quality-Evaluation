# YOUR CODE HERE
import itertools
import math

def is_square(n):
    root = int(math.isqrt(n))
    return root * root == n

def count_square_permutations(N, S):
    unique_numbers = set()
    for perm in itertools.permutations(S):
        num = int(''.join(perm))
        if is_square(num):
            unique_numbers.add(num)
    return len(unique_numbers)

# Reading input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Output the result
print(count_square_permutations(N, S))