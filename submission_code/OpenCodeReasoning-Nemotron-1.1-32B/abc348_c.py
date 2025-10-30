n = int(input().strip())
min_dict = {}
for _ in range(n):
	a, c = map(int, input().split())
	if c in min_dict:
		min_dict[c] = min(min_dict[c], a)
	else:
		min_dict[c] = a

ans = max(min_dict.values())
print(ans)