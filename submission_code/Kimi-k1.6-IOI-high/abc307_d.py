n = int(input())
s = input().strip()

stack = []
diff = [0] * (n + 2)  # Using n+2 to cover up to index n+1

for idx, c in enumerate(s):
    if c == '(':
        stack.append(idx)
    elif c == ')':
        if stack:
            i = stack.pop()
            j = idx
            diff[i] += 1
            if j + 1 <= n:
                diff[j + 1] -= 1

current = 0
result = []
for i in range(n):
    current += diff[i]
    if current == 0:
        result.append(s[i])

print(''.join(result))