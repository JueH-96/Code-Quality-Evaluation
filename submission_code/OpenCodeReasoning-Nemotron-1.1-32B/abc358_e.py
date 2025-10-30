mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	K = int(data[0])
	C = list(map(int, data[1:1+26]))
	
	max_n = 1000
	fact = [1] * (max_n + 1)
	inv_fact = [1] * (max_n + 1)
	
	for i in range(1, max_n + 1):
		fact[i] = fact[i-1] * i % mod
		
	inv_fact[max_n] = pow(fact[max_n], mod - 2, mod)
	for i in range(max_n, 0, -1):
		inv_fact[i-1] = inv_fact[i] * i % mod
		
	poly = [0] * (K + 1)
	poly[0] = 1
	
	for cap in C:
		new_poly = [0] * (K + 1)
		max_k = min(cap, K)
		for k in range(0, max_k + 1):
			for j in range(0, K - k + 1):
				new_poly[j + k] = (new_poly[j + k] + poly[j] * inv_fact[k]) % mod
		poly = new_poly
		
	ans = 0
	for L in range(1, K + 1):
		ans = (ans + poly[L] * fact[L]) % mod
		
	print(ans)

if __name__ == "__main__":
	main()