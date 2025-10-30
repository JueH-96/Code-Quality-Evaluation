# YOUR CODE HERE
S = input()
stack = []
for c in S:
    stack.append(c)
    while len(stack) >= 3 and stack[-3:] == ['A', 'B', 'C']:
        stack.pop()
        stack.pop()
        stack.pop()
print(''.join(stack))