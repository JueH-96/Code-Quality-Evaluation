# YOUR CODE HERE
def is_321_like_number(N):
    # Convert the number to a string to easily access each digit
    str_N = str(N)
    
    # Iterate over the digits and check if they are in strictly decreasing order
    for i in range(len(str_N) - 1):
        if str_N[i] <= str_N[i + 1]:
            return "No"
    
    return "Yes"

# Read input from standard input
import sys
input = sys.stdin.read
N = int(input().strip())

# Print the result
print(is_321_like_number(N))