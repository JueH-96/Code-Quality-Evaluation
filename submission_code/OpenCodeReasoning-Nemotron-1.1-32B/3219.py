from typing import List

class Solution:
	def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
		n = len(nums)
		arr = [(nums[i], i) for i in range(n)]
		arr.sort(key=lambda x: x[0])
		
		groups = []
		current_group = []
		for i in range(n):
			if i == 0:
				current_group.append(arr[i])
			else:
				if arr[i][0] - arr[i-1][0] <= limit:
					current_group.append(arr[i])
				else:
					groups.append(current_group)
					current_group = [arr[i]]
		groups.append(current_group)
		
		res = [0] * n
		for group in groups:
			indices = [idx for val, idx in group]
			indices.sort()
			for j, idx in enumerate(indices):
				res[idx] = group[j][0]
				
		return res