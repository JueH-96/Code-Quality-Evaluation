class Solution:
	def countSubarrays(self, nums: List[int], k: int) -> int:
		n = len(nums)
		next_zero_index = [[n] * n for _ in range(32)]
		
		for bit in range(32):
			last = n
			for j in range(n-1, -1, -1):
				if (nums[j] >> bit) & 1 == 0:
					last = j
				next_zero_index[bit][j] = last
		
		count = 0
		for l in range(n):
			current = nums[l]
			r = l
			while r < n:
				if current < k:
					break
				if current == k:
					count += 1
				if r == n-1:
					break
				if current == 0:
					if k == 0:
						count += (n - r - 1)
					break
				next_r = n
				for bit in range(32):
					if current & (1 << bit):
						if r+1 < n:
							candidate = next_zero_index[bit][r+1]
						else:
							candidate = n
						if candidate < next_r:
							next_r = candidate
				if next_r == n:
					if current == k:
						count += (n - r - 1)
					break
				if current == k:
					count += (next_r - r - 1)
				r = next_r
				current = current & nums[r]
		return count