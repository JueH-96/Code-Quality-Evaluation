mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	points = []
	index = 1
	for i in range(n):
		x = int(data[index])
		y = int(data[index+1])
		index += 2
		points.append((x, y))
	
	if n == 0:
		print(0)
		return
		
	order = list(range(n))
	order.sort(key=lambda i: points[i][0])
	y_vals = [points[i][1] for i in order]
	
	dp = [[0] * n for _ in range(n)]
	
	for i in range(n):
		dp[i][i] = 1
		
	for length in range(1, n):
		for i in range(0, n - length):
			j = i + length
			min_y = y_vals[i]
			max_y = y_vals[i]
			min_pos = i
			max_pos = i
			for k in range(i+1, j+1):
				if y_vals[k] < min_y:
					min_y = y_vals[k]
					min_pos = k
				if y_vals[k] > max_y:
					max_y = y_vals[k]
					max_pos = k
					
			ways = 0
			if i == min_pos:
				left_part = 1
			else:
				left_part = dp[i][min_pos-1]
			if min_pos == j:
				right_part = 1
			else:
				right_part = dp[min_pos+1][j]
			ways = (ways + left_part * right_part) % mod

			if i == max_pos:
				left_part = 1
			else:
				left_part = dp[i][max_pos-1]
			if max_pos == j:
				right_part = 1
			else:
				right_part = dp[max_pos+1][j]
			ways = (ways + left_part * right_part) % mod

			if min_pos != max_pos:
				if min_pos < max_pos:
					if i == min_pos:
						left_part = 1
					else:
						left_part = dp[i][min_pos-1]
					if max_pos == j:
						right_part = 1
					else:
						right_part = dp[max_pos+1][j]
					mid_part = 1
					if min_pos + 1 <= max_pos - 1:
						mid_part = dp[min_pos+1][max_pos-1]
					else:
						mid_part = 1
					subtract = left_part * right_part % mod * mid_part % mod
					ways = (ways - subtract + mod) % mod
				else:
					if i == max_pos:
						left_part = 1
					else:
						left_part = dp[i][max_pos-1]
					if min_pos == j:
						right_part = 1
					else:
						right_part = dp[min_pos+1][j]
					mid_part = 1
					if max_pos + 1 <= min_pos - 1:
						mid_part = dp[max_pos+1][min_pos-1]
					else:
						mid_part = 1
					subtract = left_part * right_part % mod * mid_part % mod
					ways = (ways - subtract + mod) % mod
					
			dp[i][j] = ways

	print(dp[0][n-1])

if __name__ == '__main__':
	main()