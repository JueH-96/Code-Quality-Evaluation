# YOUR CODE HERE
import sys

def dragon_string(n):
    return "L" + "o" * n + "ng"

N = int(sys.stdin.read().strip())
print(dragon_string(N))