import bisect

class Solution:
	def minOperations(self, nums: List[int]) -> int:
		max_val = 10**6
		spf = list(range(max_val + 1))
		for i in range(2, int(max_val**0.5) + 1):
			if spf[i] == i:
				for j in range(i * i, max_val + 1, i):
					if spf[j] == j:
						spf[j] = i
		
		chains = []
		for x in nums:
			chain = [x]
			current = x
			while current > 1 and spf[current] != current:
				d = spf[current]
				next_val = current // d
				chain.append(next_val)
				current = next_val
			chains.append(chain)
		
		n = len(nums)
		if n == 0:
			return 0
		
		dp = [j for j in range(len(chains[0]))]
		
		for i in range(1, n):
			prev_chain = chains[i-1]
			curr_chain = chains[i]
			m = len(prev_chain)
			M_prev = [10**15] * m
			if m > 0:
				M_prev[-1] = dp[-1]
				for k in range(m-2, -1, -1):
					M_prev[k] = min(dp[k], M_prev[k+1])
			
			rev_prev_chain = prev_chain[::-1]
			new_dp = [10**15] * len(curr_chain)
			
			for j in range(len(curr_chain)):
				val = curr_chain[j]
				pos = bisect.bisect_right(rev_prev_chain, val)
				if pos == 0:
					continue
				else:
					idx_rev = pos - 1
					k0 = m - 1 - idx_rev
					new_dp[j] = M_prev[k0] + j
			
			if min(new_dp) == 10**15:
				return -1
			dp = new_dp
		
		return min(dp)