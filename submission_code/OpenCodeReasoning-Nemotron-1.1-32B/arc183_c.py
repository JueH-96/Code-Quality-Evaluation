MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	conditions_by_x = [[] for _ in range(n)]
	index = 2
	for _ in range(m):
		l = int(data[index])
		r = int(data[index+1])
		x = int(data[index+2])
		index += 3
		l0 = l - 1
		r0 = r - 1
		x0 = x - 1
		conditions_by_x[x0].append((l0, r0))
	
	dp = [[0] * n for _ in range(n)]
	
	for i in range(n):
		bad = False
		for (l0, r0) in conditions_by_x[i]:
			if i <= l0 and r0 <= i and l0 <= i <= r0:
				bad = True
				break
		if not bad:
			dp[i][i] = 1
		else:
			dp[i][i] = 0

	for length in range(1, n):
		for i in range(0, n - length):
			j = i + length
			bad_list = [False] * (length + 1)
			for x in range(i, j + 1):
				for (l0, r0) in conditions_by_x[x]:
					if i <= l0 and r0 <= j and l0 <= x <= r0:
						bad_list[x - i] = True
						break
			total = 0
			for k in range(i, j + 1):
				if bad_list[k - i]:
					continue
				left_val = 1
				if k > i:
					left_val = dp[i][k - 1]
				right_val = 1
				if k < j:
					right_val = dp[k + 1][j]
				total = (total + left_val * right_val) % MOD
			dp[i][j] = total

	print(dp[0][n - 1] % MOD)

if __name__ == '__main__':
	main()