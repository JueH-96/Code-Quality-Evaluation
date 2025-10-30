# YOUR CODE HERE
import sys

# Read N from the first line of standard input
N = int(sys.stdin.readline())

# Read the sequence A from the second line of standard input
# Split the line by spaces and convert each part to an integer
A = list(map(int, sys.stdin.readline().split()))

# Use a list as a stack to represent the sequence of ball exponents.
# The right end of the conceptual sequence of balls corresponds to the
# end of this list (the top of the stack).
stack = []

# Process each ball based on the exponents provided in the list A
for exponent in A:
    # Add the new ball with the current exponent to the right end of the sequence
    stack.append(exponent)

    # After adding a ball, check if merges can happen at the right end.
    # Merges happen if the two rightmost balls have the same size (same exponent).
    # This process repeats as long as the condition is met.
    # The loop continues as long as the stack has at least two elements
    # and the exponent of the last element equals the exponent of the second-to-last element.
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        # If the two rightmost balls have the same exponent, they merge.
        # The rule is to remove these two balls and add a new ball with a size
        # equal to their sum. Since size is 2^k, 2^k + 2^k = 2 * 2^k = 2^(k+1).
        # So, two balls with exponent k merge into one ball with exponent k+1.
        
        # We pop the rightmost ball to get its exponent. This ball is removed.
        merged_exponent = stack.pop()
        
        # We pop the second rightmost ball. This ball is also removed.
        stack.pop()
        
        # Add the new merged ball to the right end. Its exponent is the old exponent + 1.
        stack.append(merged_exponent + 1)

# After iterating through all N balls and performing all possible merges for each,
# the number of balls remaining in the sequence is simply the number of elements left in the stack.
print(len(stack))