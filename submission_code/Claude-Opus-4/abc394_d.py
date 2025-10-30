# YOUR CODE HERE
def is_colorful_bracket_sequence(s):
    stack = []
    matching = {'(': ')', '[': ']', '<': '>'}
    
    for char in s:
        if char in matching:  # Opening bracket
            stack.append(char)
        else:  # Closing bracket
            if not stack:
                return False
            
            last_opening = stack[-1]
            expected_closing = matching[last_opening]
            
            if char == expected_closing:
                stack.pop()
            else:
                return False
    
    return len(stack) == 0

# Read input
S = input().strip()

# Check if it's a colorful bracket sequence
if is_colorful_bracket_sequence(S):
    print("Yes")
else:
    print("No")