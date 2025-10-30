import sys

S = sys.stdin.readline().strip()
mapping = {'(': ')', '[': ']', '<': '>'}
stack = []
for c in S:
	if c in mapping:
		stack.append(c)
	else:
		if not stack or mapping[stack.pop()] != c:
			print("No")
			exit(0)
print("Yes" if not stack else "No")