n, k = map(int, input().split())
s = input().strip()

blocks = []
current = 0
for c in s:
    if c == 'O':
        current += 1
    else:
        if current > 0:
            blocks.append(current)
            current = 0
if current > 0:
    blocks.append(current)

count = 0
for l in blocks:
    count += l // k

print(count)