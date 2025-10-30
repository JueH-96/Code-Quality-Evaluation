class Solution:
    def minElement(self, nums: List[int]) -> int:
        transformed = [sum(int(d) for d in str(num)) for num in nums]
        return min(transformed)