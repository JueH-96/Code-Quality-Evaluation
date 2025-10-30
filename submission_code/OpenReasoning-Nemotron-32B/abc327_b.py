B = int(input().strip())
ans = -1
for A in range(1, 16):
	power = A ** A
	if power == B:
		ans = A
		break
	if power > B:
		break
print(ans)