s = input().strip()
stack = []
for c in s:
    stack.append(c)
    while len(stack) >= 3 and stack[-3:] == ['A', 'B', 'C']:
        del stack[-3:]
print(''.join(stack))