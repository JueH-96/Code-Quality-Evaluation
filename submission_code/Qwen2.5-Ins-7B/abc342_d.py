# YOUR CODE HERE
from math import isqrt

n = int(input())
a = list(map(int, input().split()))

def is_square(x):
    return isqrt(x) ** 2 == x

count = 0
squares = set()
square_counts = [0] * (200001)

for num in a:
    if is_square(num):
        count += square_counts[num]
        squares.add(num)
    square_counts[num] += 1

print(count)