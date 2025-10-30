n = int(input())
sticks = [input().strip() for _ in range(n)]
seen = set()

for s in sticks:
    rev = s[::-1]
    key = min(s, rev)
    seen.add(key)

print(len(seen))