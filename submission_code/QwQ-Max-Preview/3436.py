class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_or = set()
        for num in nums:
            current_or = {num}
            for val in prev_or:
                current_or.add(val | num)
            # Update the minimum difference
            for val in current_or:
                diff = abs(val - k)
                if diff < min_diff:
                    min_diff = diff
                    if min_diff == 0:
                        return 0  # Early exit if possible
            prev_or = current_or
        return min_diff