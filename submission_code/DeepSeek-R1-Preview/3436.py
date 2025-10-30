class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        prev_or = set()
        min_diff = float('inf')
        for num in nums:
            current_or = set()
            current_or.add(num)
            for val in prev_or:
                current_or.add(val | num)
            for val in current_or:
                diff = abs(val - k)
                if diff < min_diff:
                    min_diff = diff
            prev_or = current_or
        return min_diff