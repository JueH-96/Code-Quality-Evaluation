n, *rest = map(int, open(0).read().split())
a = rest[:n]
stack = []
for num in a:
    stack.append(num)
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        x = stack.pop()
        stack.pop()
        stack.append(x + 1)
print(len(stack))