# YOUR CODE HERE
import sys

def is_326_like(number):
    hundreds = number // 100
    tens = (number // 10) % 10
    ones = number % 10
    return hundreds * tens == ones

def find_next_326_like(N):
    while not is_326_like(N):
        N += 1
    return N

N = int(sys.stdin.read().strip())
result = find_next_326_like(N)
print(result)