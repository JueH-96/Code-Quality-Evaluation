mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	ans = [0] * (n+1)
	ans[1] = n
	
	if n == 1:
		print("1")
		return
		
	dp = [dict() for _ in range(n)]
	
	for i in range(n):
		for j in range(i):
			d = A[i] - A[j]
			if d not in dp[i]:
				dp[i][d] = [0] * (n+1)
			dp[i][d][2] = (dp[i][d][2] + 1) % mod
			
			if d in dp[j]:
				for l in range(2, n):
					cnt = dp[j][d][l]
					if cnt != 0:
						dp[i][d][l+1] = (dp[i][d][l+1] + cnt) % mod
		
		for d, arr in dp[i].items():
			for l in range(2, n+1):
				ans[l] = (ans[l] + arr[l]) % mod
				
	res = []
	for i in range(1, n+1):
		res.append(str(ans[i]))
	print(" ".join(res))

if __name__ == "__main__":
	main()