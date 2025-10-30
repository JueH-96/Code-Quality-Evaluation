n, *rest = open(0).read().split()
n = int(n)
A = list(map(int, rest))
pos = [[] for _ in range(n+1)]
for idx, color in enumerate(A, start=1):
    pos[color].append(idx)
count = 0
for i in range(1, n+1):
    positions = pos[i]
    if abs(positions[0] - positions[1]) == 2:
        count +=1
print(count)