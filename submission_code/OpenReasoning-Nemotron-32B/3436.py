import bisect
from typing import List

class Solution:
	def minimumDifference(self, nums: List[int], k: int) -> int:
		n = len(nums)
		if n == 0:
			return 0
		
		next_occ_after = [[10**9] * n for _ in range(32)]
		positions = [[] for _ in range(32)]
		
		for i, num in enumerate(nums):
			for bit in range(32):
				if (num >> bit) & 1:
					positions[bit].append(i)
		
		for bit in range(32):
			arr = positions[bit]
			if not arr:
				continue
			for i in range(n):
				pos_index = bisect.bisect_right(arr, i)
				if pos_index < len(arr):
					next_occ_after[bit][i] = arr[pos_index]
		
		ans = 10**18
		for i in range(n):
			current = 0
			j = i
			while j < n:
				current |= nums[j]
				diff = abs(k - current)
				if diff < ans:
					ans = diff
				if ans == 0:
					break
				next_j = 10**9
				for bit in range(32):
					if (current >> bit) & 1 == 0:
						candidate = next_occ_after[bit][j]
						if candidate < next_j:
							next_j = candidate
				if next_j == 10**9:
					break
				j = next_j
		return ans