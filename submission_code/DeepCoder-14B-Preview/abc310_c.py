n = int(input())
unique = set()
for _ in range(n):
    s = input().strip()
    rev = s[::-1]
    key = min(s, rev)
    unique.add(key)
print(len(unique))