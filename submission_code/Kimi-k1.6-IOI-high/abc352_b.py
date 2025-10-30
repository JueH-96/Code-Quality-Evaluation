s = input().strip()
t = input().strip()

result = []
i = 0  # pointer for S
for j in range(len(t)):
    if i < len(s) and t[j] == s[i]:
        result.append(j + 1)  # positions are 1-based
        i += 1

print(' '.join(map(str, result)))