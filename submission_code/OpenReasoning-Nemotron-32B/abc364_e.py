import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	X = int(data[1])
	Y = int(data[2])
	dishes = []
	idx = 3
	safe_dishes = []
	for i in range(n):
		a = int(data[idx])
		b = int(data[idx + 1])
		idx += 2
		dishes.append((a, b))
		if a <= X and b <= Y:
			safe_dishes.append((a, b))
	
	n_safe = len(safe_dishes)
	dp = [[10**18] * (X + 1) for _ in range(n_safe + 1)]
	dp[0][0] = 0
	
	for a, b in safe_dishes:
		for k in range(n_safe - 1, -1, -1):
			for i in range(X, a - 1, -1):
				if dp[k][i - a] != 10**18:
					if dp[k + 1][i] > dp[k][i - a] + b:
						dp[k + 1][i] = dp[k][i - a] + b
	
	M = 0
	for k in range(n_safe + 1):
		for i in range(X + 1):
			if dp[k][i] <= Y:
				if k > M:
					M = k
	
	if M < n:
		print(M + 1)
	else:
		print(M)

if __name__ == "__main__":
	main()