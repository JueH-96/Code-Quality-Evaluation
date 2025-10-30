from collections import defaultdict

n = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

box_dict = defaultdict(list)
for a, w in zip(A, W):
    box_dict[a].append(w)

sum_total = sum(W)
sum_kept = 0

for items in box_dict.values():
    if items:
        sum_kept += max(items)

print(sum_total - sum_kept)