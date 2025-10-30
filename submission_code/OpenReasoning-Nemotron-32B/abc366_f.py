import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	k = int(data[1])
	funcs = []
	idx = 2
	for i in range(n):
		a = int(data[idx])
		b = int(data[idx + 1])
		idx += 2
		funcs.append((a, b))
	
	funcs.sort(key=lambda x: (x[0] - 1) * 1.0 / x[1])
	
	dp = [-10**30] * (k + 1)
	dp[0] = 1
	
	for a, b in funcs:
		for j in range(k, 0, -1):
			candidate = a * dp[j - 1] + b
			if candidate > dp[j]:
				dp[j] = candidate
				
	print(dp[k])

if __name__ == '__main__':
	main()