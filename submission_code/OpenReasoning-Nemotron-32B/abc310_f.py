mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	total_mod = 1
	for a in A:
		total_mod = (total_mod * a) % mod
		
	v_list = [min(a, 10) for a in A]
	
	dp = [0] * 1024
	dp[1] = 1
	
	for i in range(n):
		new_dp = [0] * 1024
		a_val = A[i]
		v_i = v_list[i]
		for mask in range(1024):
			if dp[mask] == 0:
				continue
			if a_val > 10:
				count_invalid = a_val - v_i
				new_dp[mask] = (new_dp[mask] + dp[mask] * count_invalid) % mod
				
			for k in range(1, v_i+1):
				total_new_mask = mask | (mask << k)
				if total_new_mask & 1024:
					continue
				new_mask = total_new_mask & 1023
				new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % mod
				
		dp = new_dp
		
	F_mod = sum(dp) % mod
	diff = (total_mod - F_mod) % mod
	if diff < 0:
		diff += mod
	inv_total = pow(total_mod, mod-2, mod)
	answer = diff * inv_total % mod
	print(answer)

if __name__ == '__main__':
	main()