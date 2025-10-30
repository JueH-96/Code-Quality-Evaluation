import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	M_val = int(data[1])
	wheels = []
	idx = 2
	for i in range(n):
		c = int(data[idx])
		idx += 1
		p = int(data[idx])
		idx += 1
		outcomes = list(map(int, data[idx:idx + p]))
		idx += p
		wheels.append((c, p, outcomes))
	
	if M_val == 0:
		print(0.0)
		return
		
	dp = [0.0] * M_val
	
	for x in range(M_val - 1, -1, -1):
		best = float('inf')
		for (c, p, outcomes) in wheels:
			zeros = 0
			total = 0.0
			for s in outcomes:
				if s == 0:
					zeros += 1
				else:
					next_state = x + s
					if next_state < M_val:
						total += dp[next_state]
			denom = p - zeros
			candidate = (c * p + total) / denom
			if candidate < best:
				best = candidate
		dp[x] = best
		
	print(dp[0])

if __name__ == "__main__":
	main()