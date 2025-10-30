class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        from collections import defaultdict
        counts = defaultdict(int)
        counts[0] = 1
        prefix_mod = 0
        result = 0
        for num in nums:
            A = 1 if num % modulo == k else 0
            prefix_mod = (prefix_mod + A) % modulo
            needed = (prefix_mod - k) % modulo
            result += counts.get(needed, 0)
            counts[prefix_mod] += 1
        return result