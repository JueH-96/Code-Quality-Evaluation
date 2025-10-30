import sys

# Read input from stdin
input_str = sys.stdin.readline().strip()

# Initialize stack and valid flag
stack = []
valid = True

# Iterate through each character in the string
for char in input_str:
    if char in '([<':  # Open brackets
        stack.append(char)
    elif char in ')]>':  # Close brackets
        if not stack:
            valid = False
            break
        top_char = stack[-1]
        if (char == ')' and top_char == '(') or \
           (char == ']' and top_char == '[') or \
           (char == '>' and top_char == '<'):
            stack.pop()
        else:
            valid = False
            break

# Check if the string is valid and stack is empty
if valid and len(stack) == 0:
    print("Yes")
else:
    print("No")