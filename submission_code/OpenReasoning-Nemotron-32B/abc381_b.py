s = input().strip()
n = len(s)
if n % 2 != 0:
	print("No")
else:
	for i in range(0, n, 2):
		if s[i] != s[i+1]:
			print("No")
			break
	else:
		distinct_count = len(set(s))
		if distinct_count == n // 2:
			print("Yes")
		else:
			print("No")