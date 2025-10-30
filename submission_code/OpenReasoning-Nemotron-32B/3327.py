import sys

class Solution:
	def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
		n = len(nums)
		ans = sys.maxsize
		
		for i in range(n):
			candidates = []
			if nums[i] == 1:
				for j in range(n):
					if j == i:
						continue
					if nums[j] == 1:
						cost = abs(j - i)
						candidates.append((cost, 0))
					else:
						cost = abs(j - i) + 1
						candidates.append((cost, 1))
				if len(candidates) < k - 1:
					continue
				candidates.sort(key=lambda x: x[0])
				total_cost = 0
				count_zeros = 0
				for idx in range(k - 1):
					total_cost += candidates[idx][0]
					count_zeros += candidates[idx][1]
				if count_zeros <= maxChanges:
					ans = min(ans, total_cost)
			else:
				for j in range(n):
					if j == i:
						continue
					if nums[j] == 1:
						cost = abs(j - i)
						candidates.append((cost, 0))
					else:
						cost = abs(j - i) + 1
						candidates.append((cost, 1))
				if len(candidates) < k:
					continue
				candidates.sort(key=lambda x: x[0])
				total_cost = 0
				count_zeros = 0
				for idx in range(k):
					total_cost += candidates[idx][0]
					count_zeros += candidates[idx][1]
				if count_zeros <= maxChanges:
					ans = min(ans, total_cost)
					
		return ans