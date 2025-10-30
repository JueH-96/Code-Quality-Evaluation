n = int(input().strip())
strings = []
for _ in range(n):
	strings.append(input().strip())
strings.sort(key=len)
print(''.join(strings))