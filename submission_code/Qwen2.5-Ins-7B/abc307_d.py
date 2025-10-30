# YOUR CODE HERE
n = int(input())
s = input()

stack = []
for char in s:
    if char == '(':
        stack.append(char)
    elif char == ')':
        if stack and stack[-1] == '(':
            stack.pop()
    else:
        stack.append(char)

print(''.join(stack))