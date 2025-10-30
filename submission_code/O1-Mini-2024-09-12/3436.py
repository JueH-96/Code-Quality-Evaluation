class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        prev_ors = set()
        min_diff = float('inf')
        for num in nums:
            current_ors = {num}
            for or_val in prev_ors:
                current_ors.add(or_val | num)
            for or_val in current_ors:
                min_diff = min(min_diff, abs(k - or_val))
                if min_diff == 0:
                    return 0
            prev_ors = current_ors
        return min_diff