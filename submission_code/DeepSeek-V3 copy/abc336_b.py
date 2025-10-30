# YOUR CODE HERE
def ctz(n):
    count = 0
    while n > 0 and (n & 1) == 0:
        count += 1
        n = n >> 1
    return count

# Read input
N = int(input())
# Compute and print the result
print(ctz(N))