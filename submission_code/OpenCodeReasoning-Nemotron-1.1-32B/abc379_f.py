import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	H = [0] * (n + 1)
	for i in range(1, n + 1):
		H[i] = int(next(it))
	
	next_greater = [0] * (n + 1)
	stack = []
	for i in range(n, 0, -1):
		while stack and H[stack[-1]] <= H[i]:
			stack.pop()
		if stack:
			next_greater[i] = stack[-1]
		else:
			next_greater[i] = n + 1
		stack.append(i)
	
	depth = [0] * (n + 1)
	for i in range(n, 0, -1):
		if next_greater[i] <= n:
			depth[i] = 1 + depth[next_greater[i]]
		else:
			depth[i] = 1
	
	MAX_LOG = (n).bit_length()
	dp = [[0] * (n + 1) for _ in range(MAX_LOG)]
	for i in range(1, n + 1):
		dp[0][i] = next_greater[i]
	
	for k in range(1, MAX_LOG):
		for i in range(1, n + 1):
			if dp[k - 1][i] <= n:
				dp[k][i] = dp[k - 1][dp[k - 1][i]]
			else:
				dp[k][i] = n + 1
	
	out_lines = []
	for _ in range(q):
		l = int(next(it))
		r = int(next(it))
		node = l
		for k in range(MAX_LOG - 1, -1, -1):
			if dp[k][node] <= r:
				node = dp[k][node]
		next_node = next_greater[node]
		if next_node > n:
			out_lines.append("0")
		else:
			out_lines.append(str(depth[next_node]))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()