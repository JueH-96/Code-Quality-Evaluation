# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
S = data[0]
T = data[1]

stack = []
for i, char in enumerate(T):
    if char == S[-1] and stack and stack[-1] == S[-2]:
        stack.pop()
    else:
        stack.append(char)

result = []
for i, char in enumerate(T):
    if char in stack:
        result.append(i + 1)

print(" ".join(map(str, result)))