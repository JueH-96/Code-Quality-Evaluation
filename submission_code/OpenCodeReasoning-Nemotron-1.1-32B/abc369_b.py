import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0])
	presses = []
	for i in range(1, n+1):
		parts = data[i].split()
		key = int(parts[0])
		hand = parts[1]
		presses.append((key, hand))
	
	INF = 10**18
	dp = [[0] * 101 for _ in range(101)]
	
	for (key, hand) in presses:
		new_dp = [[INF] * 101 for _ in range(101)]
		
		if hand == 'L':
			H = [INF] * 101
			for l in range(1, 101):
				f = [INF] * 101
				for r in range(1, 101):
					f[r] = dp[l][r]
				
				left_min = [INF] * 101
				for r in range(1, 101):
					if r == 1:
						left_min[r] = f[r] - r
					else:
						left_min[r] = min(left_min[r-1], f[r] - r)
				
				right_min = [INF] * 101
				for r in range(100, 0, -1):
					if r == 100:
						right_min[r] = f[r] + r
					else:
						right_min[r] = min(right_min[r+1], f[r] + r)
				
				for rp in range(1, 101):
					cost_here = min(rp + left_min[rp], -rp + right_min[rp])
					total_cost = abs(key - l) + cost_here
					if total_cost < H[rp]:
						H[rp] = total_cost
			
			for rp in range(1, 101):
				new_dp[key][rp] = H[rp]
				
		else:
			H = [INF] * 101
			for r in range(1, 101):
				f = [INF] * 101
				for l in range(1, 101):
					f[l] = dp[l][r]
				
				left_min = [INF] * 101
				for l in range(1, 101):
					if l == 1:
						left_min[l] = f[l] - l
					else:
						left_min[l] = min(left_min[l-1], f[l] - l)
				
				right_min = [INF] * 101
				for l in range(100, 0, -1):
					if l == 100:
						right_min[l] = f[l] + l
					else:
						right_min[l] = min(right_min[l+1], f[l] + l)
				
				for lp in range(1, 101):
					cost_here = min(lp + left_min[lp], -lp + right_min[lp])
					total_cost = abs(key - r) + cost_here
					if total_cost < H[lp]:
						H[lp] = total_cost
			
			for lp in range(1, 101):
				new_dp[lp][key] = H[lp]
				
		dp = new_dp
		
	ans = INF
	for l in range(1, 101):
		for r in range(1, 101):
			if dp[l][r] < ans:
				ans = dp[l][r]
				
	print(ans)

if __name__ == '__main__':
	main()