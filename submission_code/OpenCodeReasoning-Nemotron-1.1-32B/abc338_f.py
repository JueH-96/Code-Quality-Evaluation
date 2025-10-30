import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n)]
	index = 2
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		w = int(data[index+2])
		index += 3
		u -= 1
		v -= 1
		graph[u].append((v, w))
	
	total_masks = 1 << n
	INF = 10**18
	dp = [[INF] * n for _ in range(total_masks)]
	
	for i in range(n):
		dp[1 << i][i] = 0
		
	for mask in range(total_masks):
		k = bin(mask).count("1")
		if k >= 2:
			for _ in range(k-1):
				changed = False
				for v in range(n):
					if mask & (1 << v):
						for (u, w) in graph[v]:
							if mask & (1 << u):
								if dp[mask][v] + w < dp[mask][u]:
									dp[mask][u] = dp[mask][v] + w
									changed = True
				if not changed:
					break
					
		for v in range(n):
			if mask & (1 << v):
				for (u, w) in graph[v]:
					if mask & (1 << u):
						continue
					new_mask = mask | (1 << u)
					new_val = dp[mask][v] + w
					if new_val < dp[new_mask][u]:
						dp[new_mask][u] = new_val
						
	full_mask = total_masks - 1
	ans = min(dp[full_mask])
	if ans == INF:
		print("No")
	else:
		print(ans)

if __name__ == "__main__":
	main()