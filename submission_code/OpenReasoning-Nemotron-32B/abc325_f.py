import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	D = list(map(int, data[1:1+n]))
	L1 = int(data[1+n])
	C1 = int(data[2+n])
	K1 = int(data[3+n])
	L2 = int(data[4+n])
	C2 = int(data[5+n])
	K2 = int(data[6+n])
	
	INF = 10**18
	dp = [INF] * (K1 + 1)
	dp[0] = 0
	
	for d in D:
		new_dp = [INF] * (K1 + 1)
		for a_total in range(K1 + 1):
			if dp[a_total] == INF:
				continue
			a_i_max = min(K1 - a_total, (d + L1 - 1) // L1)
			for a_i in range(0, a_i_max + 1):
				if a_i * L1 >= d:
					y_i = 0
				else:
					rem = d - a_i * L1
					y_i = (rem + L2 - 1) // L2
					if y_i > K2:
						continue
				new_a = a_total + a_i
				new_y = dp[a_total] + y_i
				if new_y > K2:
					continue
				if new_y < new_dp[new_a]:
					new_dp[new_a] = new_y
		dp = new_dp
	
	ans = INF
	for a_total in range(K1 + 1):
		if dp[a_total] <= K2:
			cost = C1 * a_total + C2 * dp[a_total]
			if cost < ans:
				ans = cost
	print(-1 if ans == INF else ans)

if __name__ == "__main__":
	main()