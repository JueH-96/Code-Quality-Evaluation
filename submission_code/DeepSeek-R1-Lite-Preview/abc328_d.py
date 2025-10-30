def remove_abc(S):
    stack = []
    for char in S:
        if len(stack) >= 2 and stack[-2] == 'A' and stack[-1] == 'B' and char == 'C':
            # Remove "AB" from the stack and do not add "C"
            stack.pop()
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)

# Read input
S = input().strip()

# Get the final string after removing all "ABC"
result = remove_abc(S)

# Print the result
print(result)