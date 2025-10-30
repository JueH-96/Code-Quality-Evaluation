S = input().strip()
stack = []
matching = {')': '(', ']': '[', '>': '<'}

for char in S:
    if char in matching.values():
        stack.append(char)
    else:
        if stack and stack[-1] == matching.get(char, None):
            stack.pop()
        else:
            print("No")
            exit()
print("Yes" if not stack else "No")