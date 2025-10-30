from collections import defaultdict

n = int(input())
a = list(map(int, input().split()))

positions = defaultdict(list)
for i in range(n):
    positions[a[i]].append(i)

count = 0

for v in positions:
    pos_list = positions[v]
    if len(pos_list) >= 2:
        q = [pos - i for i, pos in enumerate(pos_list)]
        s = 0
        for j in range(len(pos_list)):
            if j > 0:
                count += j * q[j] - s
            s += q[j]

print(count)