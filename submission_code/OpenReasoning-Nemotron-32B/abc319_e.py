import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	X = int(next(it))
	Y = int(next(it))
	buses = []
	total_T = 0
	for _ in range(N-1):
		p = int(next(it))
		t_val = int(next(it))
		buses.append((p, t_val))
		total_T += t_val
		
	S = X + Y + total_T
	L = 840
	
	dp = [0] * L
	for i in range(len(buses)-1, -1, -1):
		p, t_val = buses[i]
		new_dp = [0] * L
		for r in range(L):
			w = (-r) % p
			r_next = (r + w + t_val) % L
			new_dp[r] = w + dp[r_next]
		dp = new_dp
		
	Q = int(next(it))
	out_lines = []
	for _ in range(Q):
		q = int(next(it))
		r0 = (q + X) % L
		ans = q + S + dp[r0]
		out_lines.append(str(ans))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()