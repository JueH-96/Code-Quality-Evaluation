import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	graph = [[0] * n for _ in range(n)]
	index = 1
	for i in range(n-1):
		num_weights = n - 1 - i
		weights = list(map(int, data[index:index+num_weights]))
		index += num_weights
		for j in range(i+1, n):
			graph[i][j] = weights[j - i - 1]
	
	dp = [-10**18] * (1 << n)
	dp[0] = 0
	
	for mask in range(1 << n):
		if dp[mask] == -10**18:
			continue
		for i in range(n):
			if mask & (1 << i):
				continue
			for j in range(i+1, n):
				if mask & (1 << j):
					continue
				new_mask = mask | (1 << i) | (1 << j)
				new_value = dp[mask] + graph[i][j]
				if new_value > dp[new_mask]:
					dp[new_mask] = new_value
					
	ans = max(dp)
	print(ans)

if __name__ == '__main__':
	main()