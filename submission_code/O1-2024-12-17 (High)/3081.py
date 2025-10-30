class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        i, j = 0, n // 2  # Start pointers: left half and right half
        pairs = 0
        
        # Try to pair each "smaller" element in the left half
        # with the smallest possible strictly larger element in the right half
        while i < n // 2 and j < n:
            if nums[i] < nums[j]:
                pairs += 1
                i += 1
                j += 1
            else:
                j += 1
        
        # Each valid pair removes 2 elements, so subtract 2*pairs from total length
        return n - 2 * pairs