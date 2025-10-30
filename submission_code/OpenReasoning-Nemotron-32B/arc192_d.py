mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n-1]))
	
	if n == 1:
		print(1)
		return
		
	primes_set = set()
	for a in A:
		temp = a
		f = 2
		while f * f <= temp:
			if temp % f == 0:
				primes_set.add(f)
				while temp % f == 0:
					temp //= f
			f += 1
		if temp > 1:
			primes_set.add(temp)
			
	ans = 1
	for p in primes_set:
		k_list = []
		for a in A:
			cnt = 0
			temp = a
			while temp % p == 0:
				cnt += 1
				temp //= p
			k_list.append(cnt)
		
		dp = {}
		dp[(0,0)] = 1
		
		for i in range(len(k_list)-1, -1, -1):
			new_dp = {}
			k_val = k_list[i]
			choices = []
			if k_val == 0:
				choices = [0]
			else:
				choices = [k_val, -k_val]
				
			for (s_next, m_next), count_val in dp.items():
				for x in choices:
					s_curr = x + s_next
					m_curr = min(s_curr, m_next)
					exp_val = s_curr - n * (m_curr - m_next)
					factor = pow(p, exp_val, mod)
					key = (s_curr, m_curr)
					new_dp[key] = (new_dp.get(key, 0) + count_val * factor) % mod
					
			dp = new_dp
			
		total = 0
		for val in dp.values():
			total = (total + val) % mod
			
		ans = (ans * total) % mod
		
	print(ans % mod)

if __name__ == '__main__':
	main()