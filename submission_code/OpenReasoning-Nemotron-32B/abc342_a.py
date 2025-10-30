s = input().strip()
if s[0] == s[1]:
	for i in range(2, len(s)):
		if s[i] != s[0]:
			print(i + 1)
			break
else:
	if s[0] == s[2]:
		print(2)
	else:
		print(1)