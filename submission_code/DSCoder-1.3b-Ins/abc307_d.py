N = int(input())
S = input()

stack = []
result = ''

for char in S:
    if char == '(':
        stack.append(char)
    elif char == ')':
        if stack:
            stack.pop()
            result += char
    else:
        result += char

if stack:
    print('')
else:
    print(result)