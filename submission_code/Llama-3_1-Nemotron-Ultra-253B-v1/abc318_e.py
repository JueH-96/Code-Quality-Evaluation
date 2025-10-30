from collections import defaultdict

n = int(input())
A = list(map(int, input().split()))
pos_dict = defaultdict(list)
for idx in range(n):
    x = A[idx]
    pos_dict[x].append(idx + 1)  # Using 1-based indices

total = 0

for x in pos_dict:
    pos = pos_dict[x]
    m = len(pos)
    if m < 2:
        continue
    sum_j = 0
    sum_i = 0
    for j in range(m):
        sum_j += (pos[j] - j) * j
    for i in range(m):
        sum_i += (pos[i] - i) * (m - i - 1)
    total += sum_j - sum_i

print(total)