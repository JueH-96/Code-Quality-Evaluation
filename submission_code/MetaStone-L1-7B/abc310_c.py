n = int(input())
seen = set()

for _ in range(n):
    s = input().strip()
    rev = s[::-1]
    key = min(s, rev)
    seen.add(key)

print(len(seen))