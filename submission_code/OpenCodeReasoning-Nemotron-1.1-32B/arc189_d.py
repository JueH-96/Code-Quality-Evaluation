import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	left_dp = [0] * n
	stack = []
	for i in range(n):
		left_dp[i] = A[i]
		while stack and A[stack[-1]] < left_dp[i]:
			left_dp[i] += left_dp[stack.pop()]
		stack.append(i)
	
	right_dp = [0] * n
	stack = []
	for i in range(n-1, -1, -1):
		right_dp[i] = A[i]
		while stack and A[stack[-1]] < right_dp[i]:
			right_dp[i] += right_dp[stack.pop()]
		stack.append(i)
	
	ans = [left_dp[i] + right_dp[i] - A[i] for i in range(n)]
	print(" ".join(map(str, ans)))

if __name__ == "__main__":
	main()