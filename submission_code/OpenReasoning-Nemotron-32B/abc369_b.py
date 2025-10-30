import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	n = int(data[0])
	presses = []
	for i in range(1, n+1):
		parts = data[i].split()
		a = int(parts[0]) - 1
		s = parts[1]
		presses.append((a, s))
	
	INF = 10**9
	dp = [[0] * 100 for _ in range(100)]
	
	for a, hand in presses:
		new_dp = [[INF] * 100 for _ in range(100)]
		for l in range(100):
			for r in range(100):
				if dp[l][r] == INF:
					continue
				current_cost = dp[l][r]
				if hand == 'L':
					new_l = a
					new_r = r
					move_cost = abs(a - l)
					total = current_cost + move_cost
					if total < new_dp[new_l][new_r]:
						new_dp[new_l][new_r] = total
				else:
					new_r = a
					new_l = l
					move_cost = abs(a - r)
					total = current_cost + move_cost
					if total < new_dp[new_l][new_r]:
						new_dp[new_l][new_r] = total
		dp = new_dp
	
	ans = min(min(row) for row in dp)
	print(ans)

if __name__ == '__main__':
	main()