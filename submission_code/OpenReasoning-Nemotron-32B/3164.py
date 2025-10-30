class Solution:
	def lastVisitedIntegers(self, words: List[str]) -> List[int]:
		nums = []
		consecutive_prev = 0
		result = []
		
		for word in words:
			if word == "prev":
				consecutive_prev += 1
				k = consecutive_prev
				if k > len(nums):
					result.append(-1)
				else:
					result.append(nums[len(nums) - k])
			else:
				num = int(word)
				nums.append(num)
				consecutive_prev = 0
				
		return result