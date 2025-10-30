n = int(input())
m = (n - 1).bit_length()
print(m)

friends = [[] for _ in range(m)]
for x in range(1, n + 1):
    code = x - 1
    for i in range(m):
        if code & (1 << i):
            friends[i].append(x)

for f in friends:
    print(len(f), end=' ')
    print(' '.join(map(str, f)))

s = input().strip()
binary = 0
for i in range(m):
    if s[i] == '1':
        binary += (1 << i)
x = binary + 1
print(x)