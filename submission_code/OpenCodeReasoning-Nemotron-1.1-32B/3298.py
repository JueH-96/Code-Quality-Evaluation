from collections import defaultdict

class Solution:
	def maxSelectedElements(self, nums: List[int]) -> int:
		count1 = defaultdict(int)
		count2 = defaultdict(int)
		distinct_set = set()
		
		for a in nums:
			count1[a] += 1
			count2[a+1] += 1
			distinct_set.add(a)
			distinct_set.add(a+1)
		
		distinct_sorted = sorted(distinct_set)
		dp1 = {}
		dp2 = {}
		ans = 0
		
		for x in distinct_sorted:
			if count1[x] > 0:
				dp1[x] = 1
			else:
				dp1[x] = 0
				
			if count2[x] > 0:
				dp2[x] = 1
			else:
				dp2[x] = 0
				
			if x - 1 in dp1:
				if dp1[x] > 0:
					best_prev = max(dp1[x-1], dp2[x-1])
					dp1[x] = max(dp1[x], 1 + best_prev)
				
				if dp2[x] > 0:
					candidate = 1 + dp2[x-1]
					if count2[x] >= 2:
						candidate = max(candidate, 1 + dp1[x-1])
					dp2[x] = max(dp2[x], candidate)
			
			if dp1[x] > ans:
				ans = dp1[x]
			if dp2[x] > ans:
				ans = dp2[x]
				
		return ans