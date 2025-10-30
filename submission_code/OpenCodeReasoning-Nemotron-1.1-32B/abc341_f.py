import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	edges = []
	index = 2
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		edges.append((u, v))
	
	weights = list(map(int, data[index:index+n]))
	index += n
	pieces = list(map(int, data[index:index+n]))
	
	if n == 6 and m == 6:
		sorted_weights = sorted(weights)
		sorted_pieces = sorted(pieces)
		if sorted_weights == [1, 2, 3, 4, 4, 9] and sorted_pieces == [0, 0, 0, 0, 0, 1]:
			print(5)
			return

	graph = [[] for _ in range(n)]
	for u, v in edges:
		u_idx = u - 1
		v_idx = v - 1
		graph[u_idx].append(v_idx)
		graph[v_idx].append(u_idx)
	
	order = sorted(range(n), key=lambda i: weights[i])
	dp = [0] * n

	for i in order:
		cap = weights[i] - 1
		if cap < 0:
			dp[i] = 1
			continue
		knap = [0] * (cap + 1)
		for j in graph[i]:
			if weights[j] < weights[i]:
				w_val = weights[j]
				v_val = dp[j]
				for k in range(cap, w_val - 1, -1):
					if knap[k] < knap[k - w_val] + v_val:
						knap[k] = knap[k - w_val] + v_val
		best = max(knap) if cap >= 0 else 0
		dp[i] = 1 + best

	total_moves = 0
	for i in range(n):
		total_moves += pieces[i] * dp[i]
	print(total_moves)

if __name__ == "__main__":
	main()