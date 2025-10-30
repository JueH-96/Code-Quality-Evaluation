from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev = defaultdict(int)
        for num in nums:
            curr = defaultdict(int)
            for a in prev:
                new_and = a & num
                curr[new_and] += prev[a]
            curr[num] += 1
            total += curr.get(k, 0)
            prev = curr
        return total