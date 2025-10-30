import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index])
		index += 1
		a = list(map(int, data[index:index+n]))
		index += n
		
		dp = [0] * (n + 1)
		last = [-1] * (n + 1)
		
		for i in range(1, n + 1):
			current_val = a[i - 1]
			dp[i] = dp[i - 1] + 1
			if last[current_val] != -1:
				j = last[current_val]
				candidate = dp[j] + (i - j - 1)
				if candidate < dp[i]:
					dp[i] = candidate
			last[current_val] = i - 1
		
		results.append(str(dp[n]))
	
	print("
".join(results))

if __name__ == "__main__":
	main()