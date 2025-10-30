import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	dp = [0] * (n+1)
	stack = []
	
	for i in range(n):
		if a[i] >= 0:
			if stack and stack[-1] < 0:
				option_push = dp[i] + a[i]
				option_pop = dp[i] - stack[-1]
				if option_push > option_pop:
					dp[i+1] = option_push
					stack.append(a[i])
				else:
					dp[i+1] = option_pop
					stack.pop()
			else:
				dp[i+1] = dp[i] + a[i]
				stack.append(a[i])
		else:
			if stack and stack[-1] < 0:
				option_push = dp[i] + a[i]
				option_pop = dp[i] - stack[-1]
				if option_push > option_pop:
					dp[i+1] = option_push
					stack.append(a[i])
				else:
					dp[i+1] = option_pop
					stack.pop()
			else:
				dp[i+1] = dp[i] + a[i]
				stack.append(a[i])
				
	print(dp[n])

if __name__ == "__main__":
	main()