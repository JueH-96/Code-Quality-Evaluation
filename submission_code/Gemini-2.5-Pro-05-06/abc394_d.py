import sys

def solve():
    S = sys.stdin.readline().strip()
    
    stack = []
    # Mapping of opening brackets to their corresponding closing brackets
    pairs = {
        '(': ')',
        '[': ']',
        '<': '>'
    }
    # Set of opening brackets for quick check
    opening_brackets = {'(', '[', '<'}
    
    # Iterate through each character in the input string
    for char in S:
        if char in opening_brackets:
            # If it's an opening bracket, push it onto the stack
            stack.append(char)
        else:
            # It's a closing bracket.
            # First, check if the stack is empty.
            if not stack:
                # If stack is empty, there's no opening bracket to match this closing one.
                # Example: S = ")" or S = "())" (after processing "()")
                print("No")
                return # Exit the function as it's already determined not a colorful sequence.
            
            # Pop the most recently added (innermost) opening bracket from the stack.
            last_open_char = stack.pop()
            
            # Check if the popped opening bracket correctly matches the current closing bracket.
            # pairs[last_open_char] gives the expected closing bracket for last_open_char.
            if pairs[last_open_char] != char:
                # Mismatched brackets (e.g., '(' followed by ']' or '[' followed by ')')
                # Example: S = "(]" or S = "([)]"
                print("No")
                return # Exit the function.
                
    # After iterating through all characters in S:
    # If the stack is empty, it means all opening brackets found their matching closing brackets,
    # and all closing brackets found their matches.
    if not stack:
        print("Yes")
    else:
        # If the stack is not empty, it means there are unclosed opening brackets remaining.
        # Example: S = "((" or S = "[<"
        print("No")

# This is the standard way to run the main logic in Python scripts.
if __name__ == '__main__':
    solve()