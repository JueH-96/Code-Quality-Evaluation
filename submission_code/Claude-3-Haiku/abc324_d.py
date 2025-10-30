import math

def is_square(n):
    """
    Checks if a number is a perfect square.
    """
    return int(math.sqrt(n)) ** 2 == n

def count_square_numbers(s):
    """
    Counts the number of square numbers that can be obtained by interpreting a permutation of the digits in the string s as a decimal integer.
    """
    n = len(s)
    digits = [int(d) for d in s]
    count = 0
    
    # Generate all permutations of the digits
    from itertools import permutations
    for p in permutations(range(1, n+1)):
        num = sum(digits[i-1] * 10**(n-i) for i in p)
        if is_square(num):
            count += 1
    
    return count

# Read input from stdin
n = int(input())
s = input()

# Solve the problem and print the answer
print(count_square_numbers(s))