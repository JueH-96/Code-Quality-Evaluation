# YOUR CODE HERE
import sys

B = int(sys.stdin.read().strip())

def find_integer_power(B):
    if B == 1:
        return 1
    for A in range(2, int(B**0.5) + 1):
        if A**A == B:
            return A
    return -1

result = find_integer_power(B)
print(result)