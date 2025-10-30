# YOUR CODE HERE
def find_nth_good_integer(N):
    # Even digits
    even_digits = [0, 2, 4, 6, 8]
    
    # Convert N-1 to base-5
    N -= 1
    good_integer = []
    
    if N == 0:
        good_integer.append(0)
    else:
        while N > 0:
            remainder = N % 5
            good_integer.append(even_digits[remainder])
            N //= 5
    
    # The digits are in reverse order, reverse them to get the correct number
    good_integer.reverse()
    
    # Convert list of digits to a number
    result = ''.join(map(str, good_integer))
    print(result)

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Find and print the N-th good integer
find_nth_good_integer(N)