N = int(input())
S = list(input())

stack = []

for s in S:
    if s == "(":
        stack.append(s)
    elif s == ")":
        if stack and stack[-1] == "(":
            stack.pop()
        else:
            stack.append(s)
    else:
        stack.append(s)

print("".join(stack))