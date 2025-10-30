n, m = map(int, input().split())
edges = [[] for _ in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    edges[u].append(v)
    edges[v].append(u)

W = list(map(int, input().split()))
A = list(map(int, input().split()))

# Precompute S[y] for each y
S = [[] for _ in range(n + 1)]  # S[y] will hold the list of x's for vertex y

for y in range(1, n + 1):
    wy = W[y - 1]
    xs = []
    for x in edges[y]:
        if W[x - 1] < wy:
            xs.append(x)
    # Sort by W[x] in increasing order
    xs_sorted = sorted(xs, key=lambda x: W[x - 1])
    # Compute prefix sums
    prefix = []
    total = 0
    for x in xs_sorted:
        total += W[x - 1]
        prefix.append(total)
    # Find maximum k
    max_k = 0
    for i in range(len(prefix)):
        if prefix[i] < wy:
            max_k = i + 1
        else:
            break
    S[y] = xs_sorted[:max_k]

# Sort vertices in decreasing order of W
vertices = list(range(1, n + 1))
vertices.sort(key=lambda x: -W[x - 1])

add = [0] * (n + 1)
total_ops = 0

for y in vertices:
    cnt = A[y - 1] + add[y]
    total_ops += cnt
    for x in S[y]:
        add[x] += cnt

print(total_ops)