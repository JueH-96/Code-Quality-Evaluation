from typing import List

class Solution:
	def countPairs(self, nums: List[int]) -> int:
		n = len(nums)
		if n < 2:
			return 0
		distinct_nums = set(nums)
		cache = {}
		for num in distinct_nums:
			cache[num] = self.generate_set(num)
		
		count = 0
		for i in range(n):
			for j in range(i+1, n):
				x = nums[i]
				y = nums[j]
				if y in cache[x] or x in cache[y]:
					count += 1
		return count

	def generate_set(self, a: int) -> set:
		s = str(a)
		n = len(s)
		res = set()
		res.add(a)
		if n <= 1:
			return res
		lst = list(s)
		for i in range(n):
			for j in range(i+1, n):
				new_lst = lst.copy()
				new_lst[i], new_lst[j] = new_lst[j], new_lst[i]
				num_val = int(''.join(new_lst))
				res.add(num_val)
		return res