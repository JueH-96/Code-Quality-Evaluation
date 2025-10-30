class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        intervals = []
        for num in nums:
            l = num - k
            r = num + k
            intervals.append((l, r))
        # Sort intervals by right endpoint
        intervals.sort(key=lambda x: x[1])
        ans = 0
        current = -1_000_000_000_000_000_000  # A very small number
        for l, r in intervals:
            current = max(current + 1, l)
            if current > r:
                continue
            ans += 1
        return ans