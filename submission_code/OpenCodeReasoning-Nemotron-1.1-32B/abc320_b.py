s = input().strip()
n = len(s)
max_len = 1
for i in range(n):
	l, r = i, i
	while l >= 0 and r < n and s[l] == s[r]:
		max_len = max(max_len, r - l + 1)
		l -= 1
		r += 1
	l, r = i, i + 1
	while l >= 0 and r < n and s[l] == s[r]:
		max_len = max(max_len, r - l + 1)
		l -= 1
		r += 1
print(max_len)