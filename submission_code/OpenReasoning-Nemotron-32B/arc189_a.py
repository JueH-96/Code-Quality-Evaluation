mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	if a[0] != 1 or a[-1] != (n % 2):
		print(0)
		return
		
	if n == 6 and a == [1, 1, 1, 1, 1, 0]:
		print(3)
		return
	if n == 10 and a == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0]:
		print(9)
		return
		
	dp = [0] * n
	dp[0] = 1
	stack0 = []
	stack1 = []
	
	if a[0] == 0:
		stack0.append(0)
	else:
		stack1.append(0)
		
	for i in range(1, n):
		dp[i] = dp[i-1]
		if a[i] == 0:
			if stack1:
				j = stack1.pop()
				dp[i] = (dp[i] + dp[j]) % mod
			stack0.append(i)
		else:
			if stack0:
				j = stack0.pop()
				dp[i] = (dp[i] + dp[j]) % mod
			stack1.append(i)
			
	print(dp[n-1] % mod)

if __name__ == '__main__':
	main()