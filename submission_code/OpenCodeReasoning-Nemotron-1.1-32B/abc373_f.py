import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	W = int(data[1])
	items = []
	index = 2
	for i in range(n):
		w = int(data[index])
		v = int(data[index+1])
		index += 2
		items.append((w, v))
	
	NEG_INF = -10**18
	dp = [NEG_INF] * (W+1)
	dp[0] = 0
	
	for (w_i, v_i) in items:
		new_dp = dp[:]
		for r in range(w_i):
			T = (W - r) // w_i
			if T < 0:
				continue
			g_old = []
			for t in range(T+1):
				idx = r + t * w_i
				g_old.append(dp[idx])
				
			g_new = [NEG_INF] * (T+1)
			for t in range(T+1):
				for k in range(0, t+1):
					prev_index = t - k
					if prev_index < 0:
						continue
					candidate = g_old[prev_index] + k * v_i - k * k
					if candidate > g_new[t]:
						g_new[t] = candidate
			for t in range(T+1):
				idx = r + t * w_i
				if idx <= W:
					if g_new[t] > new_dp[idx]:
						new_dp[idx] = g_new[t]
		dp = new_dp
		
	ans = max(dp)
	print(ans)

if __name__ == "__main__":
	main()