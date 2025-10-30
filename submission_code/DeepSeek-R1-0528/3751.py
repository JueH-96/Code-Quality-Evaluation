class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        base = sum(1 for num in nums if num == k)
        max_benefit = 0
        n = len(nums)
        for v in range(1, 51):
            current = 0
            best_for_v = -10**18
            for num in nums:
                a = 0
                if v != k:
                    if num == v:
                        a = 1
                    elif num == k:
                        a = -1
                current = max(a, current + a)
                if current > best_for_v:
                    best_for_v = current
            if best_for_v > max_benefit:
                max_benefit = best_for_v
        return base + max_benefit