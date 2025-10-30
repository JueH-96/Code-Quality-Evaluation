class Solution:
	def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
		n = len(nums)
		arr = [nums[i] - i for i in range(n)]
		comp = sorted(set(arr))
		comp_map = {val: idx + 1 for idx, val in enumerate(comp)}
		size = len(comp)
		
		fenw = [-10**18] * (size + 1)
		
		def update(i, val):
			while i <= size:
				if val > fenw[i]:
					fenw[i] = val
				i += i & -i
		
		def query(i):
			res = -10**18
			while i:
				if fenw[i] > res:
					res = fenw[i]
				i -= i & -i
			return res
		
		ans = -10**18
		for i in range(n):
			r = comp_map[arr[i]]
			best = query(r)
			current = nums[i] + (best if best > 0 else 0)
			if current > ans:
				ans = current
			update(r, current)
		
		return ans