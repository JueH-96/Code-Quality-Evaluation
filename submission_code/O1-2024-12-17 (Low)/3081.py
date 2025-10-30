class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll attempt to pair elements from the left half with strictly larger elements
        # from the right half. Because the array is sorted, we can do this greedily with two pointers.
        left = 0
        right = (n + 1) // 2  # start the right pointer roughly at the middle
        pairs = 0
        
        while left < (n + 1) // 2 and right < n:
            if nums[left] < nums[right]:
                pairs += 1
                left += 1
                right += 1
            else:
                # If the left element is not strictly less, move right forward
                right += 1
        
        # Each successful pair removes two elements
        return n - pairs * 2