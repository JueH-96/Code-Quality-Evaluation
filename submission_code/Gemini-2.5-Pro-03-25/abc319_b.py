# YOUR CODE HERE
import sys

# Read input N
n = int(sys.stdin.readline())

# Find divisors of N between 1 and 9
# We store them as we find them. Since we iterate j from 1 to 9,
# the list `divisors` will be inherently sorted in ascending order.
divisors = []
for j in range(1, 10):
    if n % j == 0:
        divisors.append(j) 

# Build the result string character by character
result_chars = []
# Iterate through each position i from 0 to N
for i in range(n + 1): 
    found_char = '-' # Initialize the character for position i to '-'

    # Check the condition for each potential divisor j found earlier
    # The divisors list is sorted, so the first j that satisfies the condition
    # will be the smallest required j.
    for j in divisors:
        # Calculate N/j using integer division
        n_div_j = n // j
        
        # Check if i is a multiple of N/j
        if i % n_div_j == 0:
            # If it is, we found the smallest j for this i.
            # Convert j to string and store it.
            found_char = str(j) 
            # Break the inner loop since we found the smallest j.
            break 
            
    # Append the determined character (either a digit or '-') to the list
    result_chars.append(found_char)

# Join all characters into the final result string and print it to standard output.
print("".join(result_chars))