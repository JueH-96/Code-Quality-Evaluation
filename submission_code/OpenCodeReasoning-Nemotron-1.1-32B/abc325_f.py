import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	D = list(map(int, data[1:1+n]))
	L1 = int(data[1+n])
	C1 = int(data[1+n+1])
	K1 = int(data[1+n+2])
	L2 = int(data[1+n+3])
	C2 = int(data[1+n+4])
	K2 = int(data[1+n+5])
	
	dp = {(0, 0): 0}
	
	for i in range(n):
		D_i = D[i]
		options = []
		if L2 == 0:
			max_k = 0
		else:
			max_k = (D_i + L2 - 1) // L2
		
		for k in range(0, max_k + 1):
			if k == 0:
				if L1 == 0:
					x = 0
				else:
					x = (D_i + L1 - 1) // L1
				if x <= K1:
					options.append((x, 0, x * C1))
			else:
				if D_i <= k * L2:
					if k <= K2:
						options.append((0, k, k * C2))
				else:
					if L1 == 0:
						x = 0
					else:
						remainder = D_i - k * L2
						x = (remainder + L1 - 1) // L1
					if x <= K1 and k <= K2:
						options.append((x, k, x * C1 + k * C2))
		
		new_dp = {}
		for (a, b), cost_so_far in dp.items():
			for (x, k, cost_sec) in options:
				new_a = a + x
				new_b = b + k
				if new_a <= K1 and new_b <= K2:
					new_cost = cost_so_far + cost_sec
					key = (new_a, new_b)
					if key not in new_dp or new_cost < new_dp[key]:
						new_dp[key] = new_cost
		dp = new_dp
		
	if not dp:
		print(-1)
	else:
		print(min(dp.values()))

if __name__ == '__main__':
	main()