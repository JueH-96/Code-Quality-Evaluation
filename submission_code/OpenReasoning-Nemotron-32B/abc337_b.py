s = input().strip()
n = len(s)
i = 0

while i < n and s[i] == 'A':
	i += 1

while i < n and s[i] == 'B':
	i += 1

while i < n and s[i] == 'C':
	i += 1

print("Yes" if i == n else "No")