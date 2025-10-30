class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev = set()
        for num in nums:
            curr = {num}
            for x in prev:
                curr.add(x | num)
            # Update min_diff using current OR values
            for val in curr:
                diff = abs(val - k)
                if diff < min_diff:
                    min_diff = diff
            prev = curr
        return min_diff