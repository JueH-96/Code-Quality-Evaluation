import heapq
from collections import defaultdict

class Solution:
	def maxFrequencyScore(self, nums: List[int], k: int) -> int:
		nums.sort()
		n = len(nums)
		if n == 0:
			return 0
		
		left_heap = []
		right_heap = []
		left_sum = 0
		right_sum = 0
		left_count = 0
		right_count = 0
		removed = defaultdict(int)
		
		def prune_left():
			while left_heap and removed.get(-left_heap[0], 0) > 0:
				num = -heapq.heappop(left_heap)
				removed[num] -= 1
				if removed[num] == 0:
					del removed[num]
		
		def prune_right():
			while right_heap and removed.get(right_heap[0], 0) > 0:
				num = heapq.heappop(right_heap)
				removed[num] -= 1
				if removed[num] == 0:
					del removed[num]
		
		def balance():
			prune_left()
			prune_right()
			window_size = left_count + right_count
			if window_size == 0:
				return
			target = (window_size + 1) // 2
			while left_count < target:
				if not right_count:
					break
				prune_right()
				if not right_count:
					break
				num = heapq.heappop(right_heap)
				prune_right()
				if not right_count:
					heapq.heappush(left_heap, -num)
					left_count += 1
					left_sum += num
					break
				heapq.heappush(left_heap, -num)
				left_count += 1
				left_sum += num
				right_count -= 1
				right_sum -= num
			while left_count > target:
				prune_left()
				if not left_count:
					break
				num = -heapq.heappop(left_heap)
				prune_left()
				if not left_count:
					heapq.heappush(right_heap, num)
					right_count += 1
					right_sum += num
					break
				heapq.heappush(right_heap, num)
				right_count += 1
				right_sum += num
				left_count -= 1
				left_sum -= num
			while left_count > 0 and right_count > 0:
				prune_left()
				prune_right()
				if not left_count or not right_count:
					break
				left_top = -left_heap[0]
				right_top = right_heap[0]
				if left_top > right_top:
					prune_left()
					prune_right()
					if not left_count or not right_count:
						break
					left_val = -heapq.heappop(left_heap)
					right_val = heapq.heappop(right_heap)
					left_count -= 1
					left_sum -= left_val
					right_count -= 1
					right_sum -= right_val
					heapq.heappush(left_heap, -right_val)
					left_count += 1
					left_sum += right_val
					heapq.heappush(right_heap, left_val)
					right_count += 1
					right_sum += left_val
		
		def add(num):
			nonlocal left_count, right_count, left_sum, right_sum
			if left_count == 0 or num <= -left_heap[0]:
				heapq.heappush(left_heap, -num)
				left_count += 1
				left_sum += num
			else:
				heapq.heappush(right_heap, num)
				right_count += 1
				right_sum += num
			balance()
		
		def remove(num):
			nonlocal left_count, right_count, left_sum, right_sum
			if left_count > 0 and num <= -left_heap[0]:
				removed[num] += 1
				left_count -= 1
				left_sum -= num
			else:
				removed[num] += 1
				right_count -= 1
				right_sum -= num
			balance()
		
		def get_cost():
			if left_count + right_count == 0:
				return 0
			prune_left()
			prune_right()
			if left_count == 0:
				median = right_heap[0]
			else:
				median = -left_heap[0]
			left_part = median * left_count - left_sum
			right_part = right_sum - median * right_count
			return left_part + right_part
		
		l = 0
		ans = 0
		for r in range(n):
			add(nums[r])
			while get_cost() > k and l <= r:
				remove(nums[l])
				l += 1
			ans = max(ans, r - l + 1)
		return ans