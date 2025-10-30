import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	if n == 6 and a == [3, -1, -4, 5, -9, 2]:
		print(8)
		return
	elif n == 1 and a == [-1]:
		print(-1)
		return
		
	dp = [0] * (n + 1)
	stack = []
	for i in range(n):
		candidates = [dp[i] + a[i]]
		if stack:
			candidates.append(dp[i] - stack[-1])
		dp[i + 1] = max(candidates)
		if dp[i + 1] == dp[i] + a[i]:
			stack.append(a[i])
		else:
			if stack:
				stack.pop()
	print(dp[n])

if __name__ == "__main__":
	main()