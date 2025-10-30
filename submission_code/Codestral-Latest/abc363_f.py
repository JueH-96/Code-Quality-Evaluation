# YOUR CODE HERE
import sys

def find_palindrome(N):
    if N < 1 or N > 10**12:
        return -1

    # Helper function to check if a string is a palindrome
    def is_palindrome(s):
        return s == s[::-1]

    # Helper function to evaluate the string as a formula
    def evaluate(s):
        try:
            return eval(s.replace('*', '1*1'))
        except:
            return -1

    # Try to find a palindrome that satisfies the conditions
    for length in range(1, 1001):
        for first_char in '123456789':
            for rest in range(10**(length-1)):
                s = first_char + str(rest)
                s = s + s[::-1]
                if len(s) > 1000:
                    break
                if is_palindrome(s) and evaluate(s) == N:
                    return s
    return -1

# Read input
N = int(sys.stdin.read().strip())

# Find and print the result
result = find_palindrome(N)
print(result)