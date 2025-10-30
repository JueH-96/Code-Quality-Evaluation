# YOUR CODE HERE
def construct_palindrome(N):
    if N % 2 == 1:
        # Odd length: Place a single '=' in the middle
        middle = N // 2
        result = ['-'] * N
        result[middle] = '='
    else:
        # Even length: Place two '=' in the middle
        middle1 = N // 2 - 1
        middle2 = N // 2
        result = ['-'] * N
        result[middle1] = '='
        result[middle2] = '='
    
    return ''.join(result)

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Output the result
print(construct_palindrome(N))