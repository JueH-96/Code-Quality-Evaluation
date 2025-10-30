# YOUR CODE HERE
n = int(input())
s = input()
stack = []
result = ""
for char in s:
    if char == '(':
        stack.append(char)
    elif char == ')':
        if stack and stack[-1] == '(':
            stack.pop()
        else:
            result += char
    else:
        result += char
print(result + "".join(stack))