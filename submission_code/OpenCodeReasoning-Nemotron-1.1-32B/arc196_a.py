import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	if n == 1:
		print(0)
		return
		
	dp = [0] * n
	if n >= 2:
		dp[1] = abs(a[0] - a[1])
	
	for i in range(2, n):
		dp[i] = max(dp[i-1], dp[i-2] + abs(a[i] - a[i-1]))
		
		if i >= 3:
			option1 = dp[i-3] + abs(a[i] - a[i-1]) + abs(a[i-1] - a[i-2])
			option2 = dp[i-3] + abs(a[i] - a[i-2]) + abs(a[i-1] - a[i-2])
			option3 = dp[i-3] + abs(a[i] - a[i-1]) + abs(a[i] - a[i-2])
			option4 = dp[i-3] + abs(a[i] - a[i-3]) + abs(a[i-1] - a[i-2])
			dp[i] = max(dp[i], option1, option2, option3, option4)
			
		if i >= 4:
			option5 = dp[i-4] + abs(a[i] - a[i-1]) + abs(a[i-2] - a[i-3])
			option6 = dp[i-4] + abs(a[i] - a[i-2]) + abs(a[i-1] - a[i-3])
			option7 = dp[i-4] + abs(a[i] - a[i-3]) + abs(a[i-1] - a[i-2])
			dp[i] = max(dp[i], option5, option6, option7)
			
	print(dp[n-1])

if __name__ == "__main__":
	main()