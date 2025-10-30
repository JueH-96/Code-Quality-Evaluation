class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        freq = Counter(nums)
        
        # We need the total number of "distinct slots" across both halves to be at least len(nums).
        # Each distinct value can provide up to 2 distinct slots if it appears at least twice.
        # Sum up min(count, 2) for each distinct number. If this sum >= len(nums), return True.
        
        total_slots = sum(min(v, 2) for v in freq.values())
        
        return total_slots >= len(nums)