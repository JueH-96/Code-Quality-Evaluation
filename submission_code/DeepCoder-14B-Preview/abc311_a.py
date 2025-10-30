n = int(input())
s = input().strip()

seen = set()
for i in range(len(s)):
    seen.add(s[i])
    if len(seen) == 3:
        print(i + 1)
        break