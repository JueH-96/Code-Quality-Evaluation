class Solution:
	def minimumPairRemoval(self, nums: List[int]) -> int:
		arr = nums[:]
		ops = 0
		
		while True:
			non_dec = True
			for i in range(len(arr) - 1):
				if arr[i] > arr[i+1]:
					non_dec = False
					break
			if non_dec:
				break
				
			min_sum = float('inf')
			idx = -1
			for i in range(len(arr) - 1):
				s = arr[i] + arr[i+1]
				if s < min_sum:
					min_sum = s
					idx = i
					
			arr[idx] = min_sum
			arr.pop(idx+1)
			ops += 1
			
		return ops