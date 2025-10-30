class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll use two pointers to match as many pairs as possible
        # i traverses the "left" half (potential smaller values),
        # j traverses the "right" half (potential bigger values).
        # Whenever nums[i] < nums[j], we form a valid pair and move both pointers.
        # The maximum number of pairs we can form will be removed from the array.
        
        i, j = 0, (n + 1) // 2  # start j near the middle
        pairs = 0
        
        while i < (n // 2) and j < n:
            if nums[i] < nums[j]:
                pairs += 1
                i += 1
                j += 1
            else:
                j += 1
        
        return n - 2 * pairs