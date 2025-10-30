# YOUR CODE HERE
from itertools import permutations
import math

N = int(input())
S = input().strip()

# Set to store unique square numbers
square_numbers = set()

# Generate all permutations of the string
for perm in permutations(S):
    # Convert permutation to integer
    num_str = ''.join(perm)
    num = int(num_str)
    
    # Check if it's a perfect square
    sqrt_num = int(math.sqrt(num))
    if sqrt_num * sqrt_num == num:
        square_numbers.add(num)

# Print the count of unique square numbers
print(len(square_numbers))