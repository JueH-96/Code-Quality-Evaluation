S = input().strip()
mapping = {')': '(', ']': '[', '>': '<'}
stack = []
for c in S:
	if c in mapping:
		if stack and stack[-1] == mapping[c]:
			stack.pop()
		else:
			stack.append(c)
	else:
		stack.append(c)
print("Yes" if not stack else "No")