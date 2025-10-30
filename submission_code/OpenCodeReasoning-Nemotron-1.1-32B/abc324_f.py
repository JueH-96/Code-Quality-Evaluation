import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n+1)]
	idx = 2
	for i in range(m):
		u = int(data[idx])
		v = int(data[idx+1])
		b = int(data[idx+2])
		c = int(data[idx+3])
		idx += 4
		graph[u].append((v, b, c))
	
	lo = 0.0
	hi = 10000.0
	for _ in range(60):
		mid = (lo + hi) / 2.0
		dp = [-10**18] * (n+1)
		dp[1] = 0.0
		for u in range(1, n+1):
			if dp[u] == -10**18:
				continue
			for edge in graph[u]:
				v, b, c = edge
				w = b - mid * c
				if dp[v] < dp[u] + w:
					dp[v] = dp[u] + w
					
		if dp[n] >= 0.0:
			lo = mid
		else:
			hi = mid
			
	print("{:.15f}".format(lo))

if __name__ == "__main__":
	main()