mod = 998244353
import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	first_line = data[0].split()
	N = int(first_line[0])
	M = int(first_line[1])
	S = data[1].strip()
	
	dp = [dict() for _ in range(M+1)]
	initial = tuple([0] * (N+1))
	dp[0] = {initial: 1}
	
	for i in range(M):
		for row_tuple, count_val in dp[i].items():
			for c in range(26):
				letter = chr(ord('a') + c)
				new_row = [0] * (N+1)
				for k in range(1, N+1):
					if S[k-1] == letter:
						new_row[k] = max(new_row[k-1], row_tuple[k-1] + 1)
					else:
						new_row[k] = max(new_row[k-1], row_tuple[k])
				new_row_tuple = tuple(new_row)
				if new_row_tuple in dp[i+1]:
					dp[i+1][new_row_tuple] = (dp[i+1][new_row_tuple] + count_val) % mod
				else:
					dp[i+1][new_row_tuple] = count_val % mod
					
	ans = [0] * (N+1)
	for row_tuple, count_val in dp[M].items():
		lcs_val = row_tuple[N]
		if lcs_val <= N:
			ans[lcs_val] = (ans[lcs_val] + count_val) % mod
			
	print(" ".join(str(x) for x in ans))

if __name__ == '__main__':
	main()