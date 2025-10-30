# YOUR CODE HERE
import sys

def solve():
    """
    Determines if the input string S is a colorful bracket sequence using a stack-based approach.
    A colorful bracket sequence can be reduced to an empty string by repeatedly deleting
    contiguous substrings '()', '[]', or '<>'. This reduction process is equivalent to checking
    if the string is a well-formed bracket sequence according to the rules for () [] <>.
    The standard stack algorithm for well-formed bracket sequences applies here because
    deleting a contiguous matching pair effectively removes the innermost or most recent
    pair from consideration, which is precisely what a stack models.
    """
    S = sys.stdin.readline().strip()

    # Use a list as a stack.
    stack = []
    
    # Mapping closing brackets to their corresponding opening brackets.
    mapping = {')': '(', ']': '[', '>': '<'}
    
    # Set of opening brackets for quick lookup.
    # Use mapping values as opening brackets.
    opening_brackets = set(mapping.values())

    # Iterate through the input string character by character.
    for char in S:
        if char in opening_brackets:
            # If it's an opening bracket, push it onto the stack.
            stack.append(char)
        elif char in mapping: # If it's a closing bracket
            # If the stack is empty, there's no matching opening bracket for this closing one.
            # This means a closing bracket appeared without a preceding unmatched opening one.
            if not stack:
                print("No")
                return
            
            # Pop the last encountered opening bracket from the stack.
            # In a well-formed sequence, the current closing bracket must match
            # the most recently opened bracket that is still unmatched (which is on the stack top).
            last_open = stack.pop()
            
            # Check if the popped opening bracket corresponds to the current closing bracket.
            # If they don't match, the sequence is invalid (e.g., `([)]`).
            if mapping[char] != last_open:
                print("No")
                return
        # Note: The problem constraints guarantee that the string consists only
        # of the six specified characters, so no need to handle other characters.

    # After processing the entire string:
    # If the stack is empty, it means every opening bracket found a matching closing bracket
    # in the correct order and scope, implying the string can be fully reduced to empty.
    if not stack:
        print("Yes")
    else:
        # If the stack is not empty, there are unmatched opening brackets that cannot be reduced.
        print("No")

# Call the solve function to run the logic
solve()