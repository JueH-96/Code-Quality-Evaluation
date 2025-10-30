import math

def count_positive_integers_with_9_divisors(N):
    count = 0
    for i in range(1, N+1):
        divisors = 0
        for j in range(1, int(math.sqrt(i)) + 1):
            if i % j == 0:
                if j * j == i:
                    divisors += 1
                else:
                    divisors += 2
        if divisors == 9:
            count += 1
    return count

# Read input from stdin
N = int(input())

# Solve the problem
result = count_positive_integers_with_9_divisors(N)

# Write output to stdout
print(result)