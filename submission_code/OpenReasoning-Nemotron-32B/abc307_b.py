n = int(input().strip())
words = [input().strip() for _ in range(n)]

for i in range(n):
	for j in range(n):
		if i != j:
			s = words[i] + words[j]
			if s == s[::-1]:
				print("Yes")
				exit(0)

print("No")