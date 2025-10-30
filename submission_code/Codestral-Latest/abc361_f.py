import math

def count_expressible_integers(N):
    count = 0
    for a in range(2, int(math.isqrt(N)) + 1):
        b = 2
        while a**b <= N:
            count += 1
            b += 1
    return count

# Read input
N = int(input().strip())

# Calculate and print the result
result = count_expressible_integers(N)
print(result)