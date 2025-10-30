# YOUR CODE HERE
import sys
input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

stack = []

for a in A:
    current_size = 2 ** a
    while len(stack) > 1 and stack[-1] == stack[-2]:
        stack.pop()
        stack.pop()
        current_size *= 2
    stack.append(current_size)

print(len(stack))