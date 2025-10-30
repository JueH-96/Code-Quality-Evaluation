class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        # The maximum total cost is the sum of absolute values of all elements
        # because we can always arrange the subarrays to make all terms positive.
        return sum(abs(num) for num in nums)