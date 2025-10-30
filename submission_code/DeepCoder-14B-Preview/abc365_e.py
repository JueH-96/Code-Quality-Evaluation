class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_or = set()
        for num in nums:
            current_or = {num}
            for or_val in prev_or:
                current_or.add(or_val | num)
            for or_val in current_or:
                diff = abs(k - or_val)
                if diff < min_diff:
                    min_diff = diff
            prev_or = current_or.copy()
        return min_diff