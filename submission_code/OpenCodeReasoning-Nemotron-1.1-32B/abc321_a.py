s = input().strip()
for i in range(len(s)-1):
	if s[i] <= s[i+1]:
		print("No")
		break
else:
	print("Yes")