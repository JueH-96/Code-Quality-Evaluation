# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input string and performs the specified operation
    to find the final resulting string.
    """
    # Read N, the length of the string. While not strictly required for this Python
    # implementation (since strings are dynamically sized), we read it to follow
    # the input format.
    try:
        N_str = sys.stdin.readline()
        if not N_str:
            return  # Handle empty input
        N = int(N_str)
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        return

    # `result_stack` will hold the characters of the final string.
    # It's used as a stack to build the output.
    result_stack = []
    
    # `open_paren_indices` stores the indices of '(' characters within `result_stack`.
    # This stack helps efficiently find the matching opening parenthesis for a closing one.
    open_paren_indices = []

    for char in S:
        if char == '(':
            # When an opening parenthesis is found, add it to the stack
            # and record its position.
            result_stack.append(char)
            open_paren_indices.append(len(result_stack) - 1)
        
        elif char == ')':
            # When a closing parenthesis is found, check for a matching open one.
            if open_paren_indices:
                # If `open_paren_indices` is not empty, there's a corresponding '('.
                # This pair forms the innermost scope to be deleted.
                
                # Get the index of the matching '('.
                start_index = open_paren_indices.pop()
                
                # Delete the entire segment from the result_stack starting from the
                # matching '('. This efficiently removes the `(...)` content.
                del result_stack[start_index:]
            else:
                # If there's no matching '(', this ')' is unmatched and is kept.
                result_stack.append(char)
        
        else: # The character is a lowercase letter.
            # Add the letter to the stack. It will be part of the final result
            # unless it's inside a pair that gets deleted.
            result_stack.append(char)

    # Join the characters remaining in the stack to form the final string
    # and print it to standard output.
    print("".join(result_stack))

solve()