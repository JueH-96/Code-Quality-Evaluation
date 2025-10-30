mod = 998244353

def main():
	import sys
	S = sys.stdin.readline().strip()
	n = len(S)
	
	dp = [[[[0] * 2 for _ in range(2)] for __ in range(2)] for ___ in range(2)]
	dp[0][0][0][0][0] = 1
	
	for i in range(n):
		new_dp = [[[[0] * 2 for _ in range(2)] for __ in range(2)] for ___ in range(2)]
		for a in range(2):
			for b in range(2):
				for c in range(2):
					for d in range(2):
						count = dp[a][b][c][d]
						if count == 0:
							continue
						ch = S[i]
						if ch == '?' or (ch >= 'A' and ch <= 'Z'):
							new_a = 1 if a or 1 else 0
							new_b = 1 if b or a else 0
							new_c = c
							new_d = 1 if d or c else 0
							add_val = count * 26 % mod if ch == '?' else count
							new_dp[new_a][new_b][new_c][new_d] = (new_dp[new_a][new_b][new_c][new_d] + add_val) % mod
						if ch == '?' or (ch >= 'a' and ch <= 'z'):
							new_a = a
							new_b = b
							new_c = 1 if c or b else 0
							new_d = d
							add_val = count * 26 % mod if ch == '?' else count
							new_dp[new_a][new_b][new_c][new_d] = (new_dp[new_a][new_b][new_c][new_d] + add_val) % mod
		dp = new_dp
	
	ans = 0
	for a in range(2):
		for b in range(2):
			for c in range(2):
				for d in range(2):
					if d == 0:
						ans = (ans + dp[a][b][c][d]) % mod
	print(ans)

if __name__ == '__main__':
	main()