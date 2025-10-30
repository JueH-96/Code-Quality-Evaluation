# YOUR CODE HERE
import sys

def find_326_like_number(N):
    for i in range(N, 1000):
        hundreds = i // 100
        tens = (i // 10) % 10
        ones = i % 10
        if hundreds * tens == ones:
            return i

# Read input
N = int(sys.stdin.read().strip())

# Find and print the smallest 326-like number greater than or equal to N
print(find_326_like_number(N))