class Solution:
	def numberOfSubsequences(self, nums: List[int]) -> int:
		n = len(nums)
		ans = 0
		for q in range(2, n - 2):
			left_freq = [0] * 1001
			for i in range(0, q - 1):
				num = nums[i]
				if 1 <= num <= 1000:
					left_freq[num] += 1
			
			for r in range(q + 2, n - 2):
				A = nums[q]
				B = nums[r]
				for s in range(r + 2, n):
					x = nums[s]
					product = A * x
					if product < B:
						continue
					if product > 1000 * B:
						continue
					if product % B == 0:
						target = product // B
						if 1 <= target <= 1000:
							ans += left_freq[target]
		return ans