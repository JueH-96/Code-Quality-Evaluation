n = int(input().strip())
s = input().strip()
t = input().strip()

for i in range(n):
	a = s[i]
	b = t[i]
	if a == b:
		continue
	if (a, b) in [('1','l'), ('l','1'), ('0','o'), ('o','0')]:
		continue
	print('No')
	break
else:
	print('Yes')