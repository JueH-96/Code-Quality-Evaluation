class Solution:
	def maximumEnergy(self, energy: List[int], k: int) -> int:
		n = len(energy)
		groups = [[] for _ in range(k)]
		for i in range(n):
			groups[i % k].append(energy[i])
		
		ans = -10**18
		for group in groups:
			total = 0
			max_suffix = -10**18
			for num in reversed(group):
				total += num
				if total > max_suffix:
					max_suffix = total
			if max_suffix > ans:
				ans = max_suffix
		return ans