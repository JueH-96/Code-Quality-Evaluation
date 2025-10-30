n, L, R = map(int, input().split())

size = 1 << n
query_intervals = []
stack = [(0, size - 1)]

while stack:
    s, e = stack.pop()
    if e < L or s > R:
        continue
    if s >= L and e <= R:
        query_intervals.append((s, e))
    else:
        mid = (s + e) // 2
        stack.append((mid + 1, e))
        stack.append((s, mid))

total = 0
for s, e in query_intervals:
    length = e - s + 1
    i = (length).bit_length() - 1
    j = s // length
    print(f"? {i} {j}", flush=True)
    t = int(input())
    if t == -1:
        break
    total += t
else:
    total %= 100
    print(f"! {total}", flush=True)