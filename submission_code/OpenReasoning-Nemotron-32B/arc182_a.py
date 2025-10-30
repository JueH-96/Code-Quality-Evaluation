mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	q = int(data[1])
	p = []
	v = []
	index = 2
	for i in range(q):
		p.append(int(data[index]))
		v.append(int(data[index+1]))
		index += 2
	
	dp = [0] * (n + 2)
	dp[0] = 1
	
	L = [-1] * (n + 2)
	R = [-1] * (n + 2)
	L[0] = 0
	R[n + 1] = 0
	
	for i in range(q):
		x = p[i]
		y = v[i]
		if x < 1 or x > n:
			new_dp = [0] * (n + 2)
			dp = new_dp
			continue
			
		if L[x] != -1 and L[x] > y:
			new_dp = [0] * (n + 2)
			dp = new_dp
			continue
		if R[x] != -1 and R[x] > y:
			new_dp = [0] * (n + 2)
			dp = new_dp
			continue
			
		if L[x] == -1:
			L[x] = y
		if R[x] == -1:
			R[x] = y
		if L[x] > R[x]:
			L[x], R[x] = R[x], L[x]
			
		new_dp = [0] * (n + 2)
		for j in range(n + 2):
			if dp[j] == 0:
				continue
			if L[x] <= y:
				nj = min(j, x)
				new_dp[nj] = (new_dp[nj] + dp[j]) % mod
			if R[x] <= y:
				nj = max(j, x)
				new_dp[nj] = (new_dp[nj] + dp[j]) % mod
		dp = new_dp

	ans = 0
	for i in range(n + 2):
		ans = (ans + dp[i]) % mod
	print(ans)

if __name__ == '__main__':
	main()