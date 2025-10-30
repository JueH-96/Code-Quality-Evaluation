class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        current_ors = set()
        for num in nums:
            new_ors = {num}
            for or_val in current_ors:
                new_ors.add(or_val | num)
            current_ors = new_ors
            for or_val in current_ors:
                diff = abs(or_val - k)
                if diff < min_diff:
                    min_diff = diff
        return min_diff