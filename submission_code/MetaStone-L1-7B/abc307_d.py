n = int(input())
s = list(input())
stack = []
pairs = set()

for i in range(n):
    if s[i] == '(':
        stack.append(i)
    else:
        if stack:
            j = stack.pop()
            pairs.add((j, i))

result = []
for idx in range(n):
    if idx not in pairs:
        result.append(s[idx])

print(''.join(result))