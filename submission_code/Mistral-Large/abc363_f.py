import sys
import math

def find_palindromic_expression(N):
    if N < 1 or N > 10**12:
        return -1

    # Helper function to check if a number is a palindrome
    def is_palindrome(num):
        return str(num) == str(num)[::-1]

    # Check if N itself is a palindrome
    if is_palindrome(N) and '0' not in str(N):
        return str(N)

    # Try to form a palindromic expression
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            j = N // i
            if is_palindrome(i) and is_palindrome(j) and '0' not in str(i) and '0' not in str(j):
                return f"{i}*{j}"

    # Try to form a palindromic expression with three factors
    for i in range(1, int(N**(1/3)) + 1):
        if N % i == 0:
            k = N // i
            for j in range(i, int(math.sqrt(k)) + 1):
                if k % j == 0:
                    l = k // j
                    if is_palindrome(i) and is_palindrome(j) and is_palindrome(l) and '0' not in str(i) and '0' not in str(j) and '0' not in str(l):
                        return f"{i}*{j}*{l}"

    return -1

# Read input
N = int(sys.stdin.read().strip())

# Find and print the palindromic expression
result = find_palindromic_expression(N)
print(result)