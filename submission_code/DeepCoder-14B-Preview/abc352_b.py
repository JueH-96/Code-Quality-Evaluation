s = input().strip()
t = input().strip()

i = j = 0
result = []

while i < len(s) and j < len(t):
    if s[i] == t[j]:
        result.append(j + 1)  # Positions are 1-based
        i += 1
        j += 1
    else:
        j += 1

print(' '.join(map(str, result)))