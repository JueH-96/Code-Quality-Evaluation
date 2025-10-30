# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

stack = []

for char in S:
    if char == ')':
        while stack and stack[-1] != '(':
            stack.pop()
        if stack and stack[-1] == '(':
            stack.pop()
    else:
        stack.append(char)

print(''.join(stack))