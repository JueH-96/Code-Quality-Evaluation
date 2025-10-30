n = int(input())
a_list = list(map(int, input().split()))
stack = []
for a in a_list:
    stack.append(a)
    while len(stack) >= 2 and stack[-1] == stack[-2]:
        x = stack.pop()
        stack.pop()
        stack.append(x + 1)
print(len(stack))