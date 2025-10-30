import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	edges = [[] for _ in range(n+1)]
	index = 2
	for _ in range(m):
		u = int(data[index])
		v = int(data[index+1])
		b = int(data[index+2])
		c = int(data[index+3])
		index += 4
		edges[v].append((u, b, c))
	
	low = 0.0
	high = 10000.0
	
	for _ in range(60):
		mid = (low + high) / 2.0
		dp = [-10**18] * (n+1)
		dp[1] = 0.0
		
		for v in range(1, n+1):
			for u, b, c in edges[v]:
				if dp[u] != -10**18:
					new_val = dp[u] + b - mid * c
					if new_val > dp[v]:
						dp[v] = new_val
		
		if dp[n] >= 0:
			low = mid
		else:
			high = mid
	
	ans = (low + high) / 2.0
	print("{:.15f}".format(ans))

if __name__ == '__main__':
	main()