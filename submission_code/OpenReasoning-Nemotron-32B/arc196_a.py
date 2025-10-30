import sys

def solve_even(b):
	m = len(b)
	if m == 0:
		return 0
	dp = [0] * (m + 1)
	if m >= 2:
		dp[2] = abs(b[0] - b[1])
	if m >= 4:
		for i in range(4, m + 1, 2):
			option1 = dp[i - 2] + abs(b[i - 1] - b[i - 2])
			option2 = dp[i - 4] + abs(b[i - 1] - b[i - 2]) + abs(b[i - 3] - b[i - 4])
			option3 = dp[i - 4] + abs(b[i - 1] - b[i - 3]) + abs(b[i - 2] - b[i - 4])
			option4 = dp[i - 4] + abs(b[i - 1] - b[i - 4]) + abs(b[i - 2] - b[i - 3])
			dp[i] = max(option1, option2, option3, option4)
	return dp[m]

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	a = list(map(int, data[1:1 + n]))
	
	if n % 2 == 0:
		ans = solve_even(a)
	else:
		ans1 = solve_even(a[:-1])
		ans2 = solve_even(a[1:])
		ans = max(ans1, ans2)
	print(ans)

if __name__ == '__main__':
	main()