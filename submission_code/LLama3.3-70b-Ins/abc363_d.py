def is_palindrome(n):
    """Check if a number is a palindrome."""
    return str(n) == str(n)[::-1]

def nth_smallest_palindrome(n):
    """Find the N-th smallest palindrome number."""
    count = 0
    num = 0
    while True:
        if is_palindrome(num):
            count += 1
            if count == n:
                return num
        num += 1

# Read input from stdin
n = int(input())

# Print the N-th smallest palindrome number
print(nth_smallest_palindrome(n))