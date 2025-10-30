class Solution:
    def minElement(self, nums: List[int]) -> int:
        min_val = float('inf')
        for num in nums:
            total = sum(int(d) for d in str(num))
            if total < min_val:
                min_val = total
        return min_val