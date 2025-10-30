import bisect

class Solution:
	def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
		n = len(nums)
		ones = []
		zeros = []
		for i in range(n):
			if nums[i] == 1:
				ones.append(i)
			else:
				zeros.append(i)
		total_ones = len(ones)
		L = total_ones - k
		if L < 0:
			L = 0
		
		if total_ones == 0:
			res = float('inf')
			for i in range(n):
				d_min = float('inf')
				for j in range(n):
					if j == i:
						continue
					if nums[j] == 0:
						d = abs(j - i)
						if d < d_min:
							d_min = d
				if d_min == float('inf'):
					d_min = 0
				cost = k * (1 + d_min)
				if cost < res:
					res = cost
			return res
		
		m = len(ones)
		prefix = [0] * (m + 1)
		for i in range(m):
			prefix[i + 1] = prefix[i] + ones[i]
		
		res = float('inf')
		for x in range(max(0, k - maxChanges), min(k, total_ones) + 1):
			for start in range(0, m - x + 1):
				mid_index = (start + start + x - 1) // 2
				left_count = mid_index - start
				cost_gather = left_count * ones[mid_index] - (prefix[mid_index] - prefix[start])
				right_count = start + x - 1 - mid_index
				cost_gather += (prefix[start + x] - prefix[mid_index + 1]) - right_count * ones[mid_index]
				
				leave = min(x, L, maxChanges)
				save = 0
				for j in range(leave):
					left_dist = ones[mid_index] - ones[start + j]
					right_dist = ones[start + x - 1 - j] - ones[mid_index]
					save += max(left_dist, right_dist)
				
				leave2 = k - x
				dists = []
				for z in zeros:
					dists.append(abs(z - ones[mid_index]))
				dists.sort()
				if leave2 > len(dists):
					cost_created = 10**18
				else:
					cost_created = sum(dists[:leave2])
				
				total_cost = cost_gather - save + leave + cost_created
				if total_cost < res:
					res = total_cost
		return res