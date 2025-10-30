mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n == 0:
		return
	
	ans = [0] * (n + 1)
	ans[1] = n
	
	if n == 1:
		print("1")
		return
		
	dp_prev = [[0] * n for _ in range(n)]
	for i in range(1, n):
		for j in range(i):
			dp_prev[i][j] = 1
			
	total2 = 0
	for i in range(1, n):
		for j in range(i):
			total2 = (total2 + dp_prev[i][j]) % mod
	ans[2] = total2

	for k_val in range(3, n + 1):
		dicts = [dict() for _ in range(n)]
		for j in range(n):
			d = {}
			for p in range(j):
				val = A[p]
				count_here = dp_prev[j][p]
				if count_here != 0:
					if val in d:
						d[val] = (d[val] + count_here) % mod
					else:
						d[val] = count_here
			dicts[j] = d

		dp_cur = [[0] * n for _ in range(n)]
		total_k = 0
		for i in range(n):
			for j in range(i):
				target = 2 * A[j] - A[i]
				d_j = dicts[j]
				if target in d_j:
					dp_cur[i][j] = d_j[target]
				else:
					dp_cur[i][j] = 0
				total_k = (total_k + dp_cur[i][j]) % mod
		ans[k_val] = total_k
		dp_prev = dp_cur

	res = []
	for i in range(1, n + 1):
		res.append(str(ans[i]))
	print(" ".join(res))

if __name__ == '__main__':
	main()