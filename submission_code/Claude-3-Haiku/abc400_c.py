import math

def is_good_integer(x):
    """
    Checks if a positive integer x is a good integer.
    A good integer is one that can be expressed as 2^a * b^2 for some positive integers a and b.
    """
    a = 0
    while 2 ** a <= x:
        b = int(math.sqrt(x // (2 ** a)))
        if 2 ** a * b ** 2 == x:
            return True
        a += 1
    return False

def count_good_integers(n):
    """
    Counts the number of good integers between 1 and n, inclusive.
    """
    count = 0
    for x in range(1, n+1):
        if is_good_integer(x):
            count += 1
    return count

# Read input from stdin
n = int(input())

# Solve the problem
result = count_good_integers(n)

# Write output to stdout
print(result)