class Solution:
	def minOperations(self, nums: List[int], target: int) -> int:
		cnt = [0] * 32
		for num in nums:
			k = num.bit_length() - 1
			if k < 32:
				cnt[k] += 1
		
		ans = 0
		for i in range(31):
			if target & (1 << i):
				if cnt[i] > 0:
					cnt[i] -= 1
				else:
					j = i + 1
					while j < 31 and cnt[j] == 0:
						j += 1
					if j == 31:
						return -1
					for k in range(j, i, -1):
						cnt[k] -= 1
						cnt[k-1] += 2
						ans += 1
					cnt[i] -= 1
			cnt[i+1] += cnt[i] // 2
		return ans