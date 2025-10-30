n = int(input())
a = list(map(int, input().split()))
stack = []
for num in a:
    stack.append(num)
    while len(stack) >= 2:
        x = stack[-2]
        y = stack[-1]
        if x != y:
            break
        stack.pop()
        stack.pop()
        stack.append(x + 1)
print(len(stack))