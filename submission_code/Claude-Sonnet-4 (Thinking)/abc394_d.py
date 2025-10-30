s = input().strip()

stack = []
pairs = {'(': ')', '[': ']', '<': '>'}
valid = True

for char in s:
    if char in pairs:  # Opening bracket
        stack.append(char)
    else:  # Closing bracket
        if not stack:
            valid = False
            break
        top = stack.pop()
        if pairs[top] != char:
            valid = False
            break

if valid and not stack:
    print('Yes')
else:
    print('No')