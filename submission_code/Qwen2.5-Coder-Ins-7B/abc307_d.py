# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

stack = []
result = []

for char in S:
    if char == '(':
        stack.append(char)
    elif char == ')':
        if stack and stack[-1] == '(':
            stack.pop()
        else:
            result.append(char)
    else:
        if stack:
            result.append(char)

print(''.join(result))