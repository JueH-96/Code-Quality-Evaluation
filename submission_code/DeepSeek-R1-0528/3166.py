from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        cnt = Counter(nums)
        freqs = list(cnt.values())
        min_f = min(freqs)
        
        for x in range(min_f, 0, -1):
            total = 0
            for f in freqs:
                k = (f + x) // (x + 1)
                if k * x > f:
                    break
                total += k
            else:
                return total
        
        return sum((f + 1) // 2 for f in freqs)