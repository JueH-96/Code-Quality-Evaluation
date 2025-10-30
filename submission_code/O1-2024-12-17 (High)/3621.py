class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If there's at least one value smaller than k, we can never raise it to k (operation only decreases values).
        if min(nums) < k:
            return -1
        
        # Gather all distinct values
        distinct_vals = set(nums)
        
        # The answer is simply how many distinct values are strictly greater than k
        # Each distinct value above k will require exactly one operation to remove it
        # (by picking it or k as the 'h' in the operation, so that all identical values above 'h' get reduced).
        return sum(1 for val in distinct_vals if val > k)