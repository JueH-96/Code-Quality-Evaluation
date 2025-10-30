import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, m = map(int, data[0].split())
	s = data[1].strip()
	t = data[2].strip()
	
	dp = [False] * (m + 1)
	dp[0] = True
	
	for i in range(n):
		next_dp = [False] * (m + 1)
		for j in range(m + 1):
			if not dp[j]:
				continue
			if s[i] == '#':
				if j == 0:
					next_dp[0] = True
			else:
				if j < m and t[j] == s[i]:
					next_dp[j + 1] = True
				if t[0] == s[i]:
					next_dp[1] = True
		if i >= m - 1 and dp[m]:
			next_dp[0] = True
		dp = next_dp
	
	print("Yes" if dp[0] else "No")

if __name__ == "__main__":
	main()