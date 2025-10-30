# YOUR CODE HERE
import sys

# Read the whole line from standard input and remove trailing newline
S = sys.stdin.readline().strip()

# Use a list as a stack
stack = []

# Mapping closing brackets to their corresponding opening brackets
matching = {')': '(', ']': '[', '>': '<'}

# Iterate through each character in the input string
for char in S:
    # If the character is an opening bracket
    if char in '([{<':
        # Push it onto the stack
        stack.append(char)
    # If the character is a closing bracket
    elif char in ')]>}':
        # Check if the stack is empty or if the top of the stack
        # does not match the required opening bracket for the current closing bracket
        if not stack or stack[-1] != matching[char]:
            # If either condition is true, the sequence is not colorful.
            # We can immediately print "No" and exit.
            print("No")
            # Use sys.exit() to terminate the script immediately upon finding an invalid condition
            # as per competitive programming script style.
            sys.exit()
        else:
            # If the top of the stack matches, pop it (simulating removal)
            stack.pop()
    # Characters other than the six types specified are not expected based on constraints.
    # We don't need an else block for unexpected characters.

# After processing the entire string without finding any mismatches or empty stack errors for closing brackets:
# The sequence is colorful if and only if the stack is now empty (meaning all opening brackets found a match).
if not stack:
    print("Yes")
else:
    # If the stack is not empty, there are unmatched opening brackets.
    print("No")