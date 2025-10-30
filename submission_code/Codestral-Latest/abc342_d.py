import math
from collections import defaultdict

# Read input
N = int(input())
A = list(map(int, input().split()))

# Dictionary to store the count of each square number
square_count = defaultdict(int)

# Dictionary to store the count of each non-square number
non_square_count = defaultdict(int)

# Function to check if a number is a perfect square
def is_perfect_square(x):
    return int(math.isqrt(x)) ** 2 == x

# Count the occurrences of each number in the sequence
for num in A:
    if is_perfect_square(num):
        square_count[num] += 1
    else:
        non_square_count[num] += 1

# Initialize the result
result = 0

# Calculate the number of valid pairs
for num, count in square_count.items():
    if count > 1:
        result += count * (count - 1) // 2

for num, count in non_square_count.items():
    if count > 1:
        result += count * (count - 1) // 2

# Print the result
print(result)