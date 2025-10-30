class Solution:
	def maximumEnergy(self, energy: List[int], k: int) -> int:
		n = len(energy)
		ans = -10**18
		
		for r in range(k):
			last_index = r + k * ((n - 1 - r) // k)
			cur = 0
			max_here = -10**18
			i = last_index
			while i >= r:
				cur += energy[i]
				if cur > max_here:
					max_here = cur
				i -= k
			if max_here > ans:
				ans = max_here
				
		return ans