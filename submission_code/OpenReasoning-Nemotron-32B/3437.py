import bisect
from collections import Counter
from typing import List

class Solution:
	def maximumTotalDamage(self, power: List[int]) -> int:
		if not power:
			return 0
		counter = Counter(power)
		distinct = sorted(counter.keys())
		n = len(distinct)
		total_arr = [counter[d] * d for d in distinct]
		dp = [0] * n
		
		for i in range(n):
			x = distinct[i] - 3
			pos = bisect.bisect_right(distinct, x)
			j_index = pos - 1
			
			prev_val = dp[j_index] if j_index >= 0 else 0
			
			if i == 0:
				skip_val = 0
			else:
				skip_val = dp[i-1]
				
			dp[i] = max(skip_val, total_arr[i] + prev_val)
			
		return dp[-1]