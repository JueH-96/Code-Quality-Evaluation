import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0].strip())
	s = data[1].strip()
	
	if n == 0:
		print(0)
		return
		
	dp = [-10**9] * 3
	
	if s[0] == 'R':
		dp[0] = 0
		dp[1] = 1
	elif s[0] == 'P':
		dp[1] = 0
		dp[2] = 1
	elif s[0] == 'S':
		dp[0] = 1
		dp[2] = 0
		
	for i in range(1, n):
		new_dp = [-10**9] * 3
		char = s[i]
		if char == 'R':
			new_dp[0] = max(dp[1], dp[2])
			new_dp[1] = max(dp[0], dp[2]) + 1
		elif char == 'P':
			new_dp[1] = max(dp[0], dp[2])
			new_dp[2] = max(dp[0], dp[1]) + 1
		elif char == 'S':
			new_dp[0] = max(dp[1], dp[2]) + 1
			new_dp[2] = max(dp[0], dp[1])
		dp = new_dp
		
	print(max(dp))

if __name__ == "__main__":
	main()