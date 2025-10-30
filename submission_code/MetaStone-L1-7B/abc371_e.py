from collections import defaultdict

n = int(input())
a = list(map(int, input().split()))
d = defaultdict(list)
for idx, num in enumerate(a):
    d[num].append(idx)

total = 0

for num in d:
    lst = d[num]
    for i in range(len(lst)):
        current = lst[i]
        # Find previous occurrence
        if i == 0:
            prev = -1
        else:
            prev = lst[i-1]
        # Find next occurrence
        if i == len(lst) - 1:
            next = n
        else:
            next = lst[i+1]
        contribution = (current - prev) * (next - current + 1)
        total += contribution

print(total)