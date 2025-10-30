class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter

        n = len(nums)
        half = n // 2

        # Count the frequency of each element in nums
        freq = Counter(nums)

        # Check if any element appears more than half times
        for count in freq.values():
            if count > half:
                return False

        # If no element appears more than half times, it is possible to split
        return True