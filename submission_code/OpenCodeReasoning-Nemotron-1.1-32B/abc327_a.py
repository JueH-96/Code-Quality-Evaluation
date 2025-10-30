n = int(input().strip())
s = input().strip()

for i in range(n - 1):
	if set(s[i:i+2]) == {'a', 'b'}:
		print("Yes")
		break
else:
	print("No")