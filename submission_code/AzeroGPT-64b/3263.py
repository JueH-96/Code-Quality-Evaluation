class Solution:
    def minimumCost(self, nums: List[int]) -> int:
	    a, b, c = float("inf"), float("inf"), float("inf")
	    for n in nums:
		    a, b, c = min(a, n), min(b, a), min(c, n)
		    if b < nums[0] or (b >= nums[0] and c >= nums[0]):
			    b = nums[0]
			    c = n
		    elif c >= nums[0]:
			    c = nums[0]
	    return sum([a, b, c])