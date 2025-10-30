S = input().strip()
stack = []

for c in S:
    stack.append(c)
    # Check if the last two characters are 'WA'
    while len(stack) >= 2 and stack[-2] == 'W' and stack[-1] == 'A':
        # Replace them with 'AC'
        stack.pop()
        stack.pop()
        stack.append('A')
        stack.append('C')
        
print(''.join(stack))