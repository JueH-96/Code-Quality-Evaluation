s = input().strip()
stack = []
for c in s:
    stack.append(c)
    while len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
        # Remove the last three elements
        del stack[-3:]
print(''.join(stack))