import math
from math import gcd as math_gcd
import sys

def extended_gcd(a, b):
	if b == 0:
		return (a, 1, 0)
	else:
		g, x1, y1 = extended_gcd(b, a % b)
		g, x, y = g, y1, x1 - (a // b) * y1
		return (g, x, y)

def mod_inverse(a, mod):
	g, x, y = extended_gcd(a, mod)
	if g != 1:
		return None
	else:
		return x % mod

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	P = list(map(int, data[1:1+n]))
	A = list(map(int, data[1+n:1+2*n]))
	
	P = [x-1 for x in P]
	A = [x-1 for x in A]
	
	processed = [False] * n
	cycles = []
	for i in range(n):
		if not processed[i]:
			cycle = []
			cur = i
			while not processed[cur]:
				processed[cur] = True
				cycle.append(cur)
				cur = P[cur]
			cycles.append(cycle)
	
	m = 1
	k0 = 0
	ans = [0] * n
	
	base1 = 131
	base2 = 13131
	mod1 = 10**9+7
	mod2 = 10**9+9
	
	for cycle in cycles:
		L = len(cycle)
		V = [A[i] for i in cycle]
		g = math_gcd(m, L)
		s0 = k0 % g
		
		candidates = []
		s_val = s0
		while s_val < L:
			candidates.append(s_val)
			s_val += g
		
		doubled = V + V
		n_double = 2 * L
		
		if L == 0:
			continue
			
		if len(candidates) == 1:
			best_candidate = candidates[0]
		else:
			H1 = [0] * (n_double+1)
			H2 = [0] * (n_double+1)
			power1 = [1] * (n_double+1)
			power2 = [1] * (n_double+1)
			
			for i in range(1, n_double+1):
				H1[i] = (H1[i-1] * base1 + doubled[i-1]) % mod1
				H2[i] = (H2[i-1] * base2 + doubled[i-1]) % mod2
				power1[i] = (power1[i-1] * base1) % mod1
				power2[i] = (power2[i-1] * base2) % mod2
				
			def get_hash1(l, r):
				res = (H1[r] - H1[l] * power1[r-l]) % mod1
				if res < 0:
					res += mod1
				return res
					
			def get_hash2(l, r):
				res = (H2[r] - H2[l] * power2[r-l]) % mod2
				if res < 0:
					res += mod2
				return res
					
			def cmp_substr(s1, s2, length):
				low, high = 0, length
				while low < high:
					mid = (low + high) // 2
					hash1_s1 = get_hash1(s1, s1+mid)
					hash1_s2 = get_hash1(s2, s2+mid)
					hash2_s1 = get_hash2(s1, s1+mid)
					hash2_s2 = get_hash2(s2, s2+mid)
					if hash1_s1 == hash1_s2 and hash2_s1 == hash2_s2:
						low = mid + 1
					else:
						high = mid
				if low == length:
					return 0
				if doubled[s1+low] < doubled[s2+low]:
					return -1
				else:
					return 1
					
			def less(s1, s2):
				res = cmp_substr(s1, s2, L)
				return res < 0
				
			best_candidate = candidates[0]
			for i in range(1, len(candidates)):
				if less(candidates[i], best_candidate):
					best_candidate = candidates[i]
					
		for j in range(L):
			idx = cycle[j]
			pos_in_V = (j + best_candidate) % L
			ans[idx] = V[pos_in_V]
			
		new_m = (m * L) // g
		a_val = m // g
		b_val = (best_candidate - k0) // g
		mod_val = L // g
		
		inv = mod_inverse(a_val, mod_val)
		if inv is None:
			t = 0
		else:
			t = (b_val * inv) % mod_val
			if t < 0:
				t += mod_val
		new_k0 = (k0 + t * m) % new_m
		
		m = new_m
		k0 = new_k0
		
	ans_1_indexed = [x+1 for x in ans]
	print(" ".join(map(str, ans_1_indexed)))
	
if __name__ == '__main__':
	main()