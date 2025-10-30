import sys
import functools

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])

# Define the cost function with memoization
@functools.lru_cache(None)
def cost(x):
    if x == 1:
        return 0
    else:
        return x + cost(x // 2) + cost((x + 1) // 2)

# Compute and print the result
print(cost(N))