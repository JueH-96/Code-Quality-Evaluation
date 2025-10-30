# YOUR CODE HERE
import sys

# Read input string from standard input
s = sys.stdin.readline().strip()

def is_colorful_bracket_sequence(s):
    """
    Checks if a string s is a colorful bracket sequence.
    A colorful bracket sequence can be reduced to an empty string
    by repeatedly deleting contiguous substrings "()", "[]", or "<>".
    This is equivalent to checking if the string is a well-formed
    sequence of these bracket types using a stack.
    """
    stack = []
    bracket_map = {')': '(', ']': '[', '>': '<'}
    
    # We can directly check if a character is an opening or closing bracket
    # using the values and keys of the bracket_map.

    for char in s:
        if char in bracket_map.values(): # Check if char is one of '(', '[', '<'
            # If it's an opening bracket, push it onto the stack.
            stack.append(char)
        elif char in bracket_map.keys(): # Check if char is one of ')', ']', '>'
            # If it's a closing bracket:
            # Check if the stack is empty. If so, no matching opening bracket was found.
            if not stack:
                return False
            # Pop the most recent opening bracket from the stack.
            last_open = stack.pop()
            # Check if the popped opening bracket matches the current closing bracket.
            # If not, the brackets are mismatched.
            if last_open != bracket_map[char]:
                return False
        # The problem constraint guarantees s only contains these 6 types,
        # so no other characters need handling.

    # After processing the entire string:
    # If the stack is empty, all opening brackets were matched successfully.
    # If the stack is not empty, there are unmatched opening brackets.
    return not stack

# Determine if it's a colorful bracket sequence and print the result
if is_colorful_bracket_sequence(s):
    print("Yes")
else:
    print("No")