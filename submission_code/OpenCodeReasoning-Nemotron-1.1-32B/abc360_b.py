data = input().split()
if len(data) < 2:
	print("No")
	exit(0)
S = data[0]
T = data[1]
n = len(S)

for w in range(1, n):
	for c in range(1, w + 1):
		res = []
		k = 0
		while k * w < n:
			start = k * w
			chunk_len = min(w, n - start)
			if chunk_len >= c:
				res.append(S[start + c - 1])
			k += 1
		if ''.join(res) == T:
			print("Yes")
			exit(0)

print("No")