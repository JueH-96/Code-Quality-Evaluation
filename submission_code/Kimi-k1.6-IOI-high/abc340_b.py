q = int(input())
a = []
for _ in range(q):
    parts = input().split()
    if parts[0] == '1':
        a.append(int(parts[1]))
    else:
        k = int(parts[1])
        print(a[-k])