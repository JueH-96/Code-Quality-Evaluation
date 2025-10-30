class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        res = float('inf')
        curr = set()
        for num in nums:
            curr = {num} | {num | v for v in curr}
            for v in curr:
                res = min(res, abs(k - v))
        return res