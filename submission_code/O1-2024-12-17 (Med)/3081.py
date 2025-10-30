class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll use two pointers: i scans the "left half", j scans from the "right half".
        # Because the array is sorted, we only form a pair if nums[i] < nums[j].
        # Each valid pair removes two elements. We try to maximize the number of such pairs.
        
        i, j = 0, (n + 1) // 2  # j starts roughly in the middle, i goes from the start
        pairs = 0
        while i < (n + 1) // 2 and j < n:
            if nums[i] < nums[j]:
                # We can form a pair
                pairs += 1
                i += 1
                j += 1
            else:
                # nums[i] >= nums[j], so we cannot pair i with j. Move j forward to find a bigger element.
                j += 1
        
        # Each pair removes two elements
        return n - 2 * pairs