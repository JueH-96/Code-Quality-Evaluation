class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans = float('inf')
        current_ors = set()
        for num in nums:
            next_ors = {num}
            for x in current_ors:
                next_ors.add(x | num)
            current_ors = next_ors
            for val in current_ors:
                ans = min(ans, abs(k - val))
        return ans