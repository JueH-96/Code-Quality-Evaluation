class Solution:
	def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
		n = len(source)
		m = len(pattern)
		target_set = set(targetIndices)
		
		dp_prev = [0] * (n + 1)
		
		for i in range(1, m + 1):
			dp_curr = [10**9] * (n + 1)
			for j in range(1, n + 1):
				dp_curr[j] = dp_curr[j - 1]
				if source[j - 1] == pattern[i - 1]:
					cost = 1 if (j - 1) in target_set else 0
					candidate = dp_prev[j - 1] + cost
					if candidate < dp_curr[j]:
						dp_curr[j] = candidate
			dp_prev = dp_curr
		
		k = min(dp_curr)
		return len(targetIndices) - k