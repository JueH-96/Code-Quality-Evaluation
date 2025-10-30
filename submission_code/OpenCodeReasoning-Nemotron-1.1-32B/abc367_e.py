LOG = 60

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	k = int(data[1])
	X = list(map(int, data[2:2+n]))
	A = list(map(int, data[2+n:2+2*n]))
	
	if k == 0:
		print(" ".join(map(str, A)))
		return
		
	next_node = [x-1 for x in X]
	
	dp = [[0] * n for _ in range(LOG)]
	dp[0] = next_node
	
	for j in range(1, LOG):
		for i in range(n):
			dp[j][i] = dp[j-1][dp[j-1][i]]
	
	res_index = [0] * n
	for i in range(n):
		node = i
		for j in range(LOG):
			if k & (1 << j):
				node = dp[j][node]
		res_index[i] = node
		
	result = [A[res_index[i]] for i in range(n)]
	
	print(" ".join(map(str, result)))

if __name__ == '__main__':
	main()