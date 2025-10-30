n = int(input())
s = input()
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i+1] = prefix[i] + (1 if s[i] in '()' else 0)

stack = []
removed = [False] * n

for i in range(n):
    if s[i] == '(':
        stack.append(i)
    elif s[i] == ')':
        if stack:
            j = stack.pop()
            if prefix[i] - prefix[j+1] == 0:
                removed[j] = removed[i] = True

result = []
for i in range(n):
    if not removed[i]:
        result.append(s[i])

print(''.join(result))