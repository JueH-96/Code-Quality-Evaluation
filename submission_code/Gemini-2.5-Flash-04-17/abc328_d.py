# YOUR CODE HERE
import sys

# Read the input string from standard input.
# Use strip() to remove potential trailing newline characters.
s = sys.stdin.readline().strip()

# Use a Python list as a stack. Characters that are not part of
# any removed "ABC" sequences will remain in this stack.
# This provides efficient append (O(1) amortized) and pop (O(1) amortized).
result_stack = []

# Process the input string character by character from left to right.
for char in s:
    # Always push the current character onto the stack first.
    result_stack.append(char)

    # After adding a character, check if the last three characters
    # on the stack now form the sequence "ABC".
    # We only need to check if the stack has at least 3 elements to avoid index errors.
    if len(result_stack) >= 3:
        # Check the last three elements efficiently by accessing list elements
        # using negative indices, which is an O(1) operation.
        # result_stack[-3] is the element at index len(result_stack) - 3.
        # result_stack[-2] is the element at index len(result_stack) - 2.
        # result_stack[-1] is the element at index len(result_stack) - 1 (the last element).
        if result_stack[-3] == 'A' and result_stack[-2] == 'B' and result_stack[-1] == 'C':
            # If the sequence "ABC" is found at the very end of the stack,
            # remove these three elements. This corresponds to removing
            # the leftmost "ABC" in the original string when processing
            # the string from left to right with this stack logic.
            # We need to perform 3 pop operations to remove the 'C', 'B', and 'A'.
            # Each pop from the end of a list is an efficient operation.
            result_stack.pop() # Remove the last element ('C')
            result_stack.pop() # Remove the new last element ('B')
            result_stack.pop() # Remove the new last element ('A')

# After iterating through the entire input string, the characters
# remaining in the result_stack are exactly those that are left
# after all possible "ABC" removals according to the problem rules.
# Join the characters currently in the list to form the final output string.
final_string = "".join(result_stack)

# Print the resulting string to standard output as the answer.
print(final_string)