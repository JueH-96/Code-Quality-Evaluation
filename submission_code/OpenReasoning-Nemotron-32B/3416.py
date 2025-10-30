class Solution:
	def sumDigitDifferences(self, nums: List[int]) -> int:
		n = len(nums)
		total_pairs = n * (n - 1) // 2
		d = len(str(nums[0]))
		total = 0
		
		for j in range(d):
			freq = {}
			for num in nums:
				s = str(num)
				digit = s[j]
				freq[digit] = freq.get(digit, 0) + 1
			
			same_pairs = 0
			for count in freq.values():
				same_pairs += count * (count - 1) // 2
				
			total += total_pairs - same_pairs
			
		return total