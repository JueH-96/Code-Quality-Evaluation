# YOUR CODE HERE
S = input().strip()
stack = []
for char in S:
    stack.append(char)
    if len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
        stack.pop()
        stack.pop()
        stack.pop()
print(''.join(stack))