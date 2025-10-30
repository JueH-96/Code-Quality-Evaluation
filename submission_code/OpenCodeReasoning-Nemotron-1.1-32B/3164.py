class Solution:
	def lastVisitedIntegers(self, words: List[str]) -> List[int]:
		result = []
		nums = []
		consecutive = 0
		for word in words:
			if word == "prev":
				consecutive += 1
				if consecutive <= len(nums):
					result.append(nums[-consecutive])
				else:
					result.append(-1)
			else:
				consecutive = 0
				nums.append(int(word))
		return result