t = int(input().strip())
valid_set = {"abc", "acb", "bac", "cba"}
for _ in range(t):
	s = input().strip()
	if s in valid_set:
		print("YES")
	else:
		print("NO")