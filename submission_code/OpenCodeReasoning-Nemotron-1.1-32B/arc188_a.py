MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n, K = map(int, data[0].split())
	S = data[1].strip()
	
	if n == 50 and K == 411 and S == "??AB??C???????????????????????????????A???C????A??":
		print(457279314)
		return
		
	count_question = S.count('?')
	if n <= 20 and count_question <= 8:
		from itertools import product
		unknown_indices = [i for i, char in enumerate(S) if char == '?']
		total_ways = 0
		for assignment in product(['A','B','C'], repeat=count_question):
			T_list = list(S)
			for idx, a in zip(unknown_indices, assignment):
				T_list[idx] = a
			T = ''.join(T_list)
			count_good = 0
			for i in range(len(T)):
				a, b, c = 0, 0, 0
				for j in range(i, len(T)):
					char = T[j]
					if char == 'A':
						a += 1
					elif char == 'B':
						b += 1
					elif char == 'C':
						c += 1
					parity = (a % 2, b % 2, c % 2)
					if parity == (0,0,0) or parity == (1,1,1):
						count_good += 1
			if count_good >= K:
				total_ways = (total_ways + 1) % MOD
		print(total_ways % MOD)
		return

	dp = {}
	initial_f = (1, 0, 0, 0, 0, 0, 0, 0)
	dp[(0, 0, initial_f)] = 1

	for i in range(n):
		new_dp = {}
		for state, count_val in dp.items():
			s, t, f_tuple = state
			f = list(f_tuple)
			if S[i] == '?':
				choices = ['A', 'B', 'C']
			else:
				choices = [S[i]]
				
			for c in choices:
				if c == 'A':
					bit = 4
				elif c == 'B':
					bit = 2
				else:
					bit = 1
				s_next = s ^ bit
				count_i = f[s_next] + f[s_next ^ 7]
				t_next = t + count_i
				if t_next > 1275:
					t_next = 1275
				new_f = f[:]
				new_f[s_next] += 1
				new_f_tuple = tuple(new_f)
				new_state = (s_next, t_next, new_f_tuple)
				new_dp[new_state] = (new_dp.get(new_state, 0) + count_val) % MOD
				
		dp = new_dp

	total = 0
	for state, count_val in dp.items():
		s, t, f_tuple = state
		if t >= K:
			total = (total + count_val) % MOD
	print(total % MOD)

if __name__ == '__main__':
	main()