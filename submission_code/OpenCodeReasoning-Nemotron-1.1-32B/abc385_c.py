import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	H = list(map(int, data[1:1+n]))
	groups = {}
	for i in range(n):
		h = H[i]
		if h not in groups:
			groups[h] = []
		groups[h].append(i)
	
	ans = 1
	for h, indices in groups.items():
		m = len(indices)
		if m < 2:
			continue
		dp = [dict() for _ in range(m)]
		for i in range(1, m):
			for j in range(i):
				d = indices[i] - indices[j]
				if d in dp[j]:
					dp[i][d] = dp[j][d] + 1
				else:
					dp[i][d] = 2
				if dp[i][d] > ans:
					ans = dp[i][d]
	print(ans)

if __name__ == '__main__':
	main()