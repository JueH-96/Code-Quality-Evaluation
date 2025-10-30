import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+k]))
	
	dp = [0] * (k + 1)
	if k >= 1:
		dp[1] = 0
		
	for i in range(2, k + 1):
		if i % 2 == 0:
			dp[i] = dp[i - 2] + (A[i - 1] - A[i - 2])
		else:
			dp[i] = min(dp[i - 1], dp[i - 2] + (A[i - 1] - A[i - 2]))
			
	print(dp[k])

if __name__ == '__main__':
	main()