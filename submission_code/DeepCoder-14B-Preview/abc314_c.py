n, m = map(int, input().split())
s = list(input().strip())
c = list(map(int, input().split()))

for i in range(1, m + 1):
    positions = [j for j in range(n) if c[j] == i]
    positions.sort()
    k = len(positions)
    if k <= 1:
        continue
    chars = [s[j] for j in positions]
    new_chars = [chars[-1]] + chars[:-1]
    for idx in range(k):
        s[positions[idx]] = new_chars[idx]

print(''.join(s))