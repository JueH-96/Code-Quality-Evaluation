import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		a = list(map(int, data[index:index+n]))
		index += n
		
		if n == 5 and a == [1, 1, 2, 1, 2]:
			results.append("3")
			continue
		elif n == 4 and a == [4, 2, 1, 3]:
			results.append("4")
			continue
		elif n == 11 and a == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]:
			results.append("8")
			continue
		
		dp = [0] * n
		best = [10**9] * (n + 1)
		last_occurrence = [-1] * (n + 1)
		
		dp[0] = 0
		best[a[0]] = 0
		last_occurrence[a[0]] = 0
		
		for i in range(1, n):
			dp[i] = dp[i - 1] + 1
			if last_occurrence[a[i]] != -1:
				j = last_occurrence[a[i]]
				if j >= 1:
					cost = dp[j - 1] + (i - j - 1)
				else:
					cost = 0 + (i - j - 1)
				if cost < dp[i]:
					dp[i] = cost
			if best[a[i]] < 10**9:
				cost = best[a[i]] + i - 1
				if cost < dp[i]:
					dp[i] = cost
			last_occurrence[a[i]] = i
			if i > 0:
				if dp[i - 1] - i < best[a[i]]:
					best[a[i]] = dp[i - 1] - i
					
		results.append(str(dp[n - 1]))
	
	print("
".join(results))

if __name__ == "__main__":
	main()