from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        n = len(nums)
        min_freq = min(count.values())
        
        for k in range(min_freq, 0, -1):
            groups = 0
            valid = True
            for freq in count.values():
                if (freq + k) // (k + 1) > freq // k:
                    valid = False
                    break
                groups += (freq + k) // (k + 1)
            if valid:
                return groups