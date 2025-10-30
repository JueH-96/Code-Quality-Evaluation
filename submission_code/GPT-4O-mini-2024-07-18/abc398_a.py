def generate_palindrome(N):
    if N == 1:
        return "="  # The only valid palindrome with one character is "="
    
    if N % 2 == 0:  # N is even
        # Create a palindrome with two adjacent '='
        mid = N // 2
        return '-' * (mid - 1) + '==' + '-' * (mid - 1)
    else:  # N is odd
        # Create a palindrome with one '=' in the middle
        mid = N // 2
        return '-' * mid + '=' + '-' * mid

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Generate and print the palindrome
result = generate_palindrome(N)
print(result)