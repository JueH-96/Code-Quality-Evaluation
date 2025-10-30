class Solution:
	def maximumLength(self, nums: List[int], k: int) -> int:
		n = len(nums)
		best = [dict() for _ in range(k+2)]
		ans = 0
		
		for i in range(n):
			cur = [0] * (k+2)
			
			for j in range(1, k+2):
				if j == 1:
					if nums[i] in best[j]:
						cur[j] = best[j][nums[i]] + 1
					else:
						cur[j] = 1
				else:
					candidate1 = 0
					if nums[i] in best[j]:
						candidate1 = best[j][nums[i]] + 1
					
					candidate2 = 0
					max_val = 0
					for color, val in best[j-1].items():
						if color != nums[i]:
							if val > max_val:
								max_val = val
					if max_val > 0:
						candidate2 = max_val + 1
					
					cur[j] = max(candidate1, candidate2)
				
				if cur[j] > 0:
					if nums[i] in best[j]:
						if cur[j] > best[j][nums[i]]:
							best[j][nums[i]] = cur[j]
					else:
						best[j][nums[i]] = cur[j]
				
				if cur[j] > ans:
					ans = cur[j]
					
		return ans