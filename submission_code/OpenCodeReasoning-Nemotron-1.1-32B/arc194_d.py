mod = 998244353

import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n = int(data[0].strip())
	s = data[1].strip()
	
	max_n = 5000
	fact = [1] * (max_n + 1)
	for i in range(1, max_n + 1):
		fact[i] = fact[i - 1] * i % mod

	def f(t):
		if t == "":
			return 1
		components = []
		balance = 0
		start = 0
		n_t = len(t)
		for i in range(n_t):
			if t[i] == '(':
				balance += 1
			else:
				balance -= 1
			if balance == 0:
				components.append(t[start:i + 1])
				start = i + 1
		k = len(components)
		if k >= 2:
			freq = defaultdict(int)
			for comp in components:
				freq[comp] += 1
			denom = 1
			for count in freq.values():
				denom = (denom * fact[count]) % mod
			arrangements = fact[k] * pow(denom, mod - 2, mod) % mod
			res = arrangements
			for comp in components:
				res = (res * f(comp)) % mod
			return res
		else:
			return f(t[1:-1])
	
	result = f(s)
	print(result)

if __name__ == '__main__':
	main()