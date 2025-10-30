class Solution:
	def countTestedDevices(self, batteryPercentages: List[int]) -> int:
		tests_so_far = 0
		count = 0
		for b in batteryPercentages:
			if b - tests_so_far > 0:
				count += 1
				tests_so_far += 1
		return count