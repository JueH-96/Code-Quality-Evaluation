n = int(input().strip())
dishes = [input().strip() for _ in range(n)]

flag = True
for i in range(1, n):
	if dishes[i] == 'sweet' and dishes[i-1] == 'sweet':
		if i < n - 1:
			flag = False
		break

print("Yes" if flag else "No")