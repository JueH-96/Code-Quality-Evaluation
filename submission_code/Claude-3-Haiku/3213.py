from collections import Counter

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)):
            freq = Counter()
            for j in range(i, len(nums)):
                freq[nums[j]] += 1
                if freq[nums[j]] >= k:
                    count += 1
        return count