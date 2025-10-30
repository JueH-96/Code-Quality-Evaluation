n = int(input())
A = list(map(int, input().split()))
count = [0, 0, 0, 0]

for a in A:
    deltas = [0, 1, 3]
    s = set()
    for delta in deltas:
        r = (delta - a) % 4
        s.add(r)
    for r in s:
        count[r] += 1

max_count = max(count)
print(max_count)