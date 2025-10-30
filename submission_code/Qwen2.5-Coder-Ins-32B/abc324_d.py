# YOUR CODE HERE
from itertools import permutations
import math

def is_square(num):
    root = int(math.sqrt(num))
    return root * root == num

def count_square_permutations(N, S):
    unique_permutations = set(int(''.join(p)) for p in permutations(S))
    count = sum(1 for num in unique_permutations if is_square(num))
    return count

import sys
input = sys.stdin.read().split()
N = int(input[0])
S = input[1]
print(count_square_permutations(N, S))