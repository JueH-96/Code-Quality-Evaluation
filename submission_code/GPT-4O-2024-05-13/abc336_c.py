# YOUR CODE HERE
def nth_good_integer(N):
    # Define the digits that can be used in a good integer
    even_digits = ['0', '2', '4', '6', '8']
    
    # Initialize the result as an empty string
    result = ''
    
    # Convert N to zero-based index
    N -= 1
    
    # Generate the N-th good integer
    while N >= 0:
        result = even_digits[N % 5] + result
        N = N // 5 - 1
    
    return result

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Print the N-th smallest good integer
print(nth_good_integer(N))