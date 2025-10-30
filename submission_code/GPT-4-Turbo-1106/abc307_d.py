import sys

def remove_parentheses(s):
    stack = []
    for char in s:
        if char == ')' and stack and stack[-1] == '(':
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)

# Read input
N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

# Process and output the result
result = remove_parentheses(S)
print(result)