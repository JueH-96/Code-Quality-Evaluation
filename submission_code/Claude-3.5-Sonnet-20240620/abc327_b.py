# YOUR CODE HERE
import math

def find_a(b):
    for a in range(1, int(math.log2(b)) + 2):
        if a ** a == b:
            return a
    return -1

b = int(input())
result = find_a(b)
print(result)