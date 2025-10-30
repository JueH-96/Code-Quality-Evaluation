import sys

def main():
	N = int(sys.stdin.readline().strip())
	if N < 2:
		print(0)
		return
	
	S = set([N])
	stack = [N]
	while stack:
		x = stack.pop()
		if x < 2:
			continue
		a = x // 2
		b = (x + 1) // 2
		for child in (a, b):
			if child not in S:
				S.add(child)
				stack.append(child)
	
	sorted_S = sorted(S)
	dp = {}
	for n in sorted_S:
		if n < 2:
			dp[n] = 0
		else:
			a_val = n // 2
			b_val = (n + 1) // 2
			dp[n] = n + dp[a_val] + dp[b_val]
	
	print(dp[N])

if __name__ == "__main__":
	main()