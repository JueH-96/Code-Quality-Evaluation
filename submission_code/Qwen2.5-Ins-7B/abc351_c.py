# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

stack = []
for size in a:
    while len(stack) >= 2 and stack[-1] == stack[-2] and stack[-1] == size:
        stack.pop()
        stack.pop()
        stack.append(stack[-1] * 2)
    stack.append(size)

print(len(stack))