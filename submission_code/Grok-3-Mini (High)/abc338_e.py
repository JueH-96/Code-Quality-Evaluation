import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
chords = []
for _ in range(N):
    A = int(data[index])
    B = int(data[index + 1])
    index += 2
    min_ab = min(A, B)
    max_ab = max(A, B)
    chords.append((min_ab, max_ab))
chords.sort()
size = 2 * N
tree = [0] * (size + 1)
def add(pos, val):
    while pos <= size:
        tree[pos] += val
        pos += pos & -pos
def prefix_sum(pos):
    res = 0
    while pos > 0:
        res += tree[pos]
        pos -= pos & -pos
    return res
def range_sum(L, R):
    if L > R:
        return 0
    return prefix_sum(R) - prefix_sum(L - 1)
for min_ab, max_ab in chords:
    A = min_ab
    B = max_ab
    L = A + 1
    R = B - 1
    if L <= R and range_sum(L, R) > 0:
        print("Yes")
        sys.exit()
    add(B, 1)
print("No")