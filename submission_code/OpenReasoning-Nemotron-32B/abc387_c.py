import sys
from functools import lru_cache

def count_up_to(n):
	if n < 10:
		return 0
	s = str(n)
	n_len = len(s)
	
	@lru_cache(maxsize=None)
	def dfs(pos, tight, state_flag, first_digit):
		if pos == n_len:
			return 1 if state_flag == 2 else 0
		total = 0
		limit = int(s[pos]) if tight else 9
		for d in range(0, limit + 1):
			next_tight = tight and (d == limit)
			if state_flag == 0:
				if d == 0:
					total += dfs(pos + 1, next_tight, 0, 0)
				else:
					total += dfs(pos + 1, next_tight, 1, d)
			elif state_flag == 1:
				if d < first_digit:
					total += dfs(pos + 1, next_tight, 2, first_digit)
			else:
				if d < first_digit:
					total += dfs(pos + 1, next_tight, 2, first_digit)
		return total
	return dfs(0, True, 0, 0)

def main():
	data = sys.stdin.readline().split()
	L = int(data[0])
	R = int(data[1])
	count_R = count_up_to(R)
	count_L_minus = count_up_to(L - 1)
	result = count_R - count_L_minus
	print(result)

if __name__ == "__main__":
	main()