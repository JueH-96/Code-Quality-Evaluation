n = int(input().strip())
dishes = [input().strip() for _ in range(n)]

for i in range(n - 1):
	if dishes[i] == 'sweet' and dishes[i + 1] == 'sweet':
		if i + 1 != n - 1:
			print('No')
			break
else:
	print('Yes')