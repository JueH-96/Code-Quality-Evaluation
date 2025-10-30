import heapq
from collections import defaultdict

class Solution:
	def minOperations(self, nums: List[int], x: int, k: int) -> int:
		n = len(nums)
		if k == 0:
			return 0
		if n < k * x:
			return 0
		
		C = [0] * (n - x + 1)
		left_heap = []
		right_heap = []
		left_removed = defaultdict(int)
		right_removed = defaultdict(int)
		left_size = 0
		right_size = 0
		left_sum = 0
		right_sum = 0
		
		def add(val):
			nonlocal left_size, right_size, left_sum, right_sum
			if not left_heap or val <= -left_heap[0]:
				heapq.heappush(left_heap, -val)
				left_size += 1
				left_sum += val
			else:
				heapq.heappush(right_heap, val)
				right_size += 1
				right_sum += val
			rebalance()
		
		def remove(val):
			nonlocal left_size, right_size, left_sum, right_sum
			if left_heap and val <= -left_heap[0]:
				left_removed[val] += 1
				left_size -= 1
				left_sum -= val
			else:
				right_removed[val] += 1
				right_size -= 1
				right_sum -= val
			rebalance()
		
		def rebalance():
			nonlocal left_size, right_size
			target_left = (x + 1) // 2
			while left_size < target_left and right_heap:
				while right_heap and right_removed.get(right_heap[0], 0) > 0:
					val = heapq.heappop(right_heap)
					right_removed[val] -= 1
					if right_removed[val] == 0:
						del right_removed[val]
				if not right_heap:
					break
				val = heapq.heappop(right_heap)
				heapq.heappush(left_heap, -val)
				left_size += 1
				left_sum += val
				right_size -= 1
				right_sum -= val
			while left_size > target_left and left_heap:
				while left_heap and left_removed.get(-left_heap[0], 0) > 0:
					stored_val = heapq.heappop(left_heap)
					actual_val = -stored_val
					left_removed[actual_val] -= 1
					if left_removed[actual_val] == 0:
						del left_removed[actual_val]
				if not left_heap:
					break
				stored_val = heapq.heappop(left_heap)
				actual_val = -stored_val
				heapq.heappush(right_heap, actual_val)
				right_size += 1
				right_sum += actual_val
				left_size -= 1
				left_sum -= actual_val
		
		def compute_cost():
			while left_heap and left_removed.get(-left_heap[0], 0) > 0:
				stored_val = heapq.heappop(left_heap)
				actual_val = -stored_val
				left_removed[actual_val] -= 1
				if left_removed[actual_val] == 0:
					del left_removed[actual_val]
			while right_heap and right_removed.get(right_heap[0], 0) > 0:
				val = heapq.heappop(right_heap)
				right_removed[val] -= 1
				if right_removed[val] == 0:
					del right_removed[val]
			if not left_heap:
				return 0
			median = -left_heap[0]
			cost_left = left_size * median - left_sum
			cost_right = right_sum - right_size * median
			return cost_left + cost_right
		
		for i in range(x):
			add(nums[i])
		C[0] = compute_cost()
		
		for i in range(1, n - x + 1):
			remove(nums[i - 1])
			add(nums[i + x - 1])
			C[i] = compute_cost()
		
		INF = 10**18
		dp = [[INF] * (k + 1) for _ in range(n + 1)]
		dp[0][0] = 0
		for i in range(1, n + 1):
			for j in range(0, k + 1):
				dp[i][j] = dp[i - 1][j]
				if i >= x and j >= 1:
					candidate = dp[i - x][j - 1] + C[i - x]
					if candidate < dp[i][j]:
						dp[i][j] = candidate
		return dp[n][k]