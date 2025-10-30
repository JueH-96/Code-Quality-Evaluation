import sys
sys.setrecursionlimit(1000000)

data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + M]))

# Build segment tree
tree = [1000000005] * (4 * N)  # Initialize with a large number larger than max A_i

def build(node, start, end):
    if start == end:
        tree[node] = A[start]
    else:
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        tree[node] = min(tree[2 * node], tree[2 * node + 1])

build(1, 0, N - 1)

def query(node, start, end, val):
    if tree[node] > val:
        return -1
    if start == end:
        return start
    mid = (start + end) // 2
    res = query(2 * node, start, mid, val)
    if res != -1:
        return res
    return query(2 * node + 1, mid + 1, end, val)

for b_val in B:
    idx = query(1, 0, N - 1, b_val)
    if idx == -1:
        print(-1)
    else:
        print(idx + 1)