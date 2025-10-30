import bisect
from collections import defaultdict

class Solution:
	def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
		n = len(nums)
		pos_dict = defaultdict(list)
		for index, num in enumerate(nums):
			pos_dict[num].append(index)
		
		res = []
		for q in queries:
			val = nums[q]
			arr = pos_dict[val]
			if len(arr) == 1:
				res.append(-1)
			else:
				pos = bisect.bisect_left(arr, q)
				if pos == 0:
					left_candidate = arr[-1]
				else:
					left_candidate = arr[pos-1]
				if pos == len(arr)-1:
					right_candidate = arr[0]
				else:
					right_candidate = arr[pos+1]
				
				d_left = min(abs(q - left_candidate), n - abs(q - left_candidate))
				d_right = min(abs(q - right_candidate), n - abs(q - right_candidate))
				res.append(min(d_left, d_right))
		return res