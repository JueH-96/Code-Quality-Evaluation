def is_colorful_bracket_sequence(s):
    stack = []
    for char in s:
        # Add the current character to the stack
        stack.append(char)
        
        # Check if the last two characters in the stack form a valid bracket pair
        # If they do, remove them (simulating the deletion operation)
        while len(stack) >= 2 and ((stack[-2] == '(' and stack[-1] == ')') or
                                   (stack[-2] == '[' and stack[-1] == ']') or
                                   (stack[-2] == '<' and stack[-1] == '>')):
            stack.pop()  # Remove the closing bracket
            stack.pop()  # Remove the opening bracket
    
    # If the stack is empty, we were able to reduce S to an empty string
    return len(stack) == 0

# Read input
s = input().strip()

# Check if the string is a colorful bracket sequence
if is_colorful_bracket_sequence(s):
    print("Yes")
else:
    print("No")