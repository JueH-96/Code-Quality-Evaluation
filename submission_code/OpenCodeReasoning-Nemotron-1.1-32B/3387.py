class Solution:
	def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
		n = len(nums)
		count_less = 0
		count_greater = 0
		list_less = []
		list_greater = []
		for num in nums:
			if num < k:
				count_less += 1
				list_less.append(k - num)
			elif num > k:
				count_greater += 1
				list_greater.append(num - k)
		
		max_greater = n - n//2 - 1
		changes_greater = max(0, count_greater - max_greater)
		changes_less = max(0, count_less - n//2)
		
		list_greater.sort()
		list_less.sort()
		
		total_cost = sum(list_greater[:changes_greater]) + sum(list_less[:changes_less])
		return total_cost