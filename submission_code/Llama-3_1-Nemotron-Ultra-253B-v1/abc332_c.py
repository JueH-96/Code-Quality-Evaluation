n, m = map(int, input().split())
s = input().strip()

blocks = []
current = []
for c in s:
    if c == '0':
        if current:
            blocks.append(current)
            current = []
    else:
        current.append(c)
if current:
    blocks.append(current)

max_logo = 0
for block in blocks:
    count_1 = sum(1 for x in block if x == '1')
    count_2 = sum(1 for x in block if x == '2')
    required = count_2 + max(0, count_1 - m)
    if required > max_logo:
        max_logo = required

print(max_logo)