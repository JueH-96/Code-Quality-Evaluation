import heapq

class Solution:
	def minOperations(self, nums: List[int], x: int, k: int) -> int:
		n = len(nums)
		if k == 0:
			return 0
		
		lower = []
		upper = []
		sum_lower = 0
		sum_upper = 0
		size_lower = 0
		size_upper = 0
		heap_id = [None] * n
		deleted = set()
		cost_arr = [0] * n
		
		def clean_heaps():
			nonlocal lower, upper
			while lower and lower[0][1] in deleted:
				heapq.heappop(lower)
			while upper and upper[0][1] in deleted:
				heapq.heappop(upper)
		
		def balance_heaps(desired_size_lower, desired_size_upper):
			nonlocal lower, upper, sum_lower, sum_upper, size_lower, size_upper, heap_id, deleted
			while size_lower < desired_size_lower and upper:
				clean_heaps()
				if not upper:
					break
				v, idx = heapq.heappop(upper)
				sum_upper -= v
				size_upper -= 1
				heapq.heappush(lower, (-v, idx))
				sum_lower += v
				size_lower += 1
				heap_id[idx] = 'lower'
			
			while size_lower > desired_size_lower and lower:
				clean_heaps()
				if not lower:
					break
				neg_v, idx = heapq.heappop(lower)
				v = -neg_v
				sum_lower -= v
				size_lower -= 1
				heapq.heappush(upper, (v, idx))
				sum_upper += v
				size_upper += 1
				heap_id[idx] = 'upper'
			
			while True:
				clean_heaps()
				if not lower or not upper:
					break
				top_lower = -lower[0][0]
				top_upper = upper[0][0]
				if top_lower <= top_upper:
					break
				neg_v_low, idx_low = heapq.heappop(lower)
				v_low = -neg_v_low
				v_high, idx_high = heapq.heappop(upper)
				heapq.heappush(lower, (-v_high, idx_high))
				heapq.heappush(upper, (v_low, idx_low))
				sum_lower = sum_lower - v_low + v_high
				sum_upper = sum_upper - v_high + v_low
				heap_id[idx_low] = 'upper'
				heap_id[idx_high] = 'lower'
		
		for idx in range(x):
			v = nums[idx]
			if not lower:
				heapq.heappush(lower, (-v, idx))
				sum_lower += v
				heap_id[idx] = 'lower'
				size_lower += 1
			else:
				clean_heaps()
				current_median = -lower[0][0]
				if v <= current_median:
					heapq.heappush(lower, (-v, idx))
					sum_lower += v
					heap_id[idx] = 'lower'
					size_lower += 1
				else:
					heapq.heappush(upper, (v, idx))
					sum_upper += v
					heap_id[idx] = 'upper'
					size_upper += 1
		
		desired_size_lower = (x + 1) // 2
		desired_size_upper = x // 2
		balance_heaps(desired_size_lower, desired_size_upper)
		clean_heaps()
		median = -lower[0][0] if lower else 0
		cost_arr[x-1] = (sum_upper - sum_lower) + median * (size_lower - size_upper)
		
		for i in range(x, n):
			idx_remove = i - x
			v_remove = nums[idx_remove]
			deleted.add(idx_remove)
			if heap_id[idx_remove] == 'lower':
				sum_lower -= v_remove
				size_lower -= 1
			else:
				sum_upper -= v_remove
				size_upper -= 1
			
			v_add = nums[i]
			clean_heaps()
			if not lower:
				heapq.heappush(lower, (-v_add, i))
				sum_lower += v_add
				heap_id[i] = 'lower'
				size_lower += 1
			else:
				current_median = -lower[0][0]
				if v_add <= current_median:
					heapq.heappush(lower, (-v_add, i))
					sum_lower += v_add
					heap_id[i] = 'lower'
					size_lower += 1
				else:
					heapq.heappush(upper, (v_add, i))
					sum_upper += v_add
					heap_id[i] = 'upper'
					size_upper += 1
			
			desired_size_lower = (x + 1) // 2
			desired_size_upper = x // 2
			balance_heaps(desired_size_lower, desired_size_upper)
			clean_heaps()
			median = -lower[0][0] if lower else 0
			cost_arr[i] = (sum_upper - sum_lower) + median * (size_lower - size_upper)
		
		dp = [[10**18] * (k+1) for _ in range(n)]
		for i in range(n):
			for j in range(k+1):
				if i >= 1:
					dp[i][j] = dp[i-1][j]
				else:
					if j == 0:
						dp[i][j] = 0
					else:
						dp[i][j] = 10**18
				
				if j >= 1 and i >= x-1:
					prev_index = i - x
					prev_val = 0
					if prev_index >= 0:
						prev_val = dp[prev_index][j-1]
					else:
						if j-1 == 0:
							prev_val = 0
						else:
							prev_val = 10**18
					candidate = prev_val + cost_arr[i]
					if candidate < dp[i][j]:
						dp[i][j] = candidate
		
		ans = min(dp[i][k] for i in range(n))
		return ans