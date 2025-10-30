import math

n = int(input())
m = math.ceil(math.log2(n))
print(m)

assignments = [[] for _ in range(m)]
for bottle in range(1, n + 1):
    for friend in range(m):
        if (bottle - 1) >> friend & 1:
            assignments[friend].append(bottle)

for friend_bottles in assignments:
    print(len(friend_bottles), *sorted(friend_bottles))

s = input()
spoiled_bottle_index = int(s, 2)
print(spoiled_bottle_index + 1)