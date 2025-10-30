class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        current_ors = set()
        
        for num in nums:
            new_ors = set()
            new_ors.add(num)
            for or_val in current_ors:
                new_or = or_val | num
                new_ors.add(new_or)
            for new_or in new_ors:
                diff = abs(k - new_or)
                if diff < min_diff:
                    min_diff = diff
                if diff == 0:
                    return 0
            current_ors = new_ors
        return min_diff