s = input().strip()
if len(s) != 6:
	print("No")
elif s[:3] != "ABC":
	print("No")
elif not s[3:].isdigit():
	print("No")
else:
	num = int(s[3:])
	if 1 <= num <= 349 and num != 316:
		print("Yes")
	else:
		print("No")