n = int(input())
s = input().strip()
seen = set()
for i in range(n):
    c = s[i]
    seen.add(c)
    if len(seen) == 3:
        print(i + 1)
        break