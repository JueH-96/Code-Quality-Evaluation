n = int(input().strip())
strings = [input().strip() for _ in range(n)]

for i in range(n):
	for j in range(n):
		if i != j:
			s = strings[i] + strings[j]
			if s == s[::-1]:
				print("Yes")
				exit(0)

print("No")