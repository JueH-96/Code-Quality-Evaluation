# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
stack = []
for e in A:
    stack.append(e)
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        e = stack.pop()
        stack.pop()
        stack.append(e + 1)
print(len(stack))