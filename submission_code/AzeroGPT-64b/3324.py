from collections import Counter
class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        num_counts = Counter(nums)
        can_split = True

        for num, count in num_counts.items():
            if count > 1:
                can_split = False
                break

        return can_split