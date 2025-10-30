n = int(input())
a = list(map(int, input().split()))

stack = []
for i in range(n):
    stack.append(a[i])
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        stack.pop()
        stack.pop()
        if len(stack) == 0:
            break
        stack.append(stack[-1] + 1)
    if len(stack) == 0:
        stack.append(a[i])

print(len(stack))