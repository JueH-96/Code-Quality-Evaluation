# YOUR CODE HERE
import itertools
import math

def is_square(n):
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

def count_square_permutations(N, S):
    unique_squares = set()
    for perm in itertools.permutations(S):
        # Convert tuple to string, then to integer
        num_str = ''.join(perm)
        # Skip numbers with leading zeros
        if num_str[0] == '0':
            continue
        num = int(num_str)
        if is_square(num):
            unique_squares.add(num)
    return len(unique_squares)

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
S = data[1]

# Calculate and print the result
result = count_square_permutations(N, S)
print(result)