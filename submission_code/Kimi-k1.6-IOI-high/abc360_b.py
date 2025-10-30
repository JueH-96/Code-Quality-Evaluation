s, t = input().split()
n = len(s)
found = False

for w in range(1, n):
    chunks = [s[i:i+w] for i in range(0, n, w)]
    for c in range(1, w + 1):
        current = []
        for chunk in chunks:
            if len(chunk) >= c:
                current.append(chunk[c-1])
        if ''.join(current) == t:
            found = True
            break
    if found:
        break

print("Yes" if found else "No")