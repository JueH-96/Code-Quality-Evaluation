import sys
sys.setrecursionlimit(10000)

def f(n):
	if n < 10:
		return 0
	s = str(n)
	n_len = len(s)
	total = 0
	for d in range(2, n_len):
		for a in range(1, 10):
			total += a ** (d - 1)
	
	from functools import lru_cache
	@lru_cache(maxsize=None)
	def dfs(pos, tight, first_digit):
		if pos == n_len:
			return 1
		res = 0
		if pos == 0:
			low_bound = 1
			high_bound = int(s[0]) if tight else 9
			for d in range(low_bound, high_bound + 1):
				new_tight = tight and (d == int(s[0]))
				res += dfs(pos + 1, new_tight, d)
			return res
		else:
			low_bound = 0
			high_bound = first_digit - 1
			if tight:
				high_bound = min(high_bound, int(s[pos]))
			if high_bound < low_bound:
				return 0
			for d in range(low_bound, high_bound + 1):
				new_tight = tight and (d == int(s[pos]))
				res += dfs(pos + 1, new_tight, first_digit)
			return res
	
	total += dfs(0, True, 0)
	return total

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	L = int(data[0])
	R = int(data[1])
	res = f(R) - f(L - 1)
	print(res)

if __name__ == "__main__":
	main()