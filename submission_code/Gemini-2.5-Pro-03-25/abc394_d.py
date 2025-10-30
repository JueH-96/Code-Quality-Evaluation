# YOUR CODE HERE
import sys

# Function implementing the solution logic
def solve():
    # Read the input string from stdin
    # .strip() removes potential leading/trailing whitespace including the newline character
    S = sys.stdin.readline().strip()
    
    # Initialize an empty list to use as a stack
    # The stack will store unmatched opening brackets encountered so far.
    stack = []
    
    # Dictionary mapping closing brackets to their corresponding opening brackets
    # This helps quickly check if a closing bracket matches the top of the stack.
    match_map = {')': '(', ']': '[', '>': '<'}
    
    # Set of opening brackets for efficient checking if a character is an opening bracket
    opening_brackets = {'(', '[', '<'}
    
    # Set of closing brackets for efficient checking if a character is a closing bracket
    # This check could also be done by seeing if `char` is in `match_map.keys()`.
    closing_brackets = {')', ']', '>'} 
    
    # Iterate through each character in the input string S
    for char in S:
        # If the character is an opening bracket
        if char in opening_brackets:
            # Push it onto the stack. It needs a matching closing bracket later.
            stack.append(char)
        # If the character is a closing bracket
        elif char in closing_brackets:
            # Check if the stack is empty. 
            # If the stack is empty, we encountered a closing bracket without a corresponding opening bracket.
            # This means the string is improperly formed.
            if not stack:
                # The string is not a colorful bracket sequence. Print No and exit.
                print("No")
                # Using return to exit the function early prevents further processing.
                return 
            
            # If stack is not empty, pop the top element.
            # This element represents the most recently encountered unmatched opening bracket.
            top_char = stack.pop()
            
            # Check if the popped opening bracket corresponds to the current closing bracket.
            # We use match_map to find the expected opening bracket for the current closing bracket `char`.
            # For example, if `char` is ')', `match_map[char]` is '('. We check if `top_char` is '('.
            if top_char != match_map[char]:
                # If the popped bracket doesn't match the expected one based on the current closing bracket, 
                # the sequence is mismatched (e.g., encountering ']' when '(' is expected).
                # The string is not a colorful bracket sequence. Print No and exit.
                print("No")
                # Exit the function early.
                return 
        
        # According to the problem statement, the input string S will only contain 
        # the six specified types of characters: (, ), [, ], <, >.
        # Therefore, we do not need an 'else' case to handle any other character types.
            
    # After processing all characters in the string S:
    # We need to check if the stack is empty.
    # An empty stack means all opening brackets found their matching closing brackets in the correct order.
    if not stack:
        # The string is a colorful bracket sequence.
        print("Yes")
    else:
        # If the stack is not empty, it means there are unmatched opening brackets remaining.
        # These brackets were never closed.
        # The string is not a colorful bracket sequence.
        print("No")

# The standard pattern in Python to ensure the main logic runs only when the script is executed directly.
# This is good practice, especially for larger programs or modules.
if __name__ == '__main__':
    solve()