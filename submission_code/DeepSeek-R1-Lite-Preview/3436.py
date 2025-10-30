class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_or_set = set()
        for num in nums:
            current_or_set = set()
            current_or_set.add(num)
            for or_val in prev_or_set:
                current_or_set.add(or_val | num)
            prev_or_set = current_or_set
            for or_val in current_or_set:
                diff = abs(or_val - k)
                if diff < min_diff:
                    min_diff = diff
                    if min_diff == 0:
                        return 0
        return min_diff