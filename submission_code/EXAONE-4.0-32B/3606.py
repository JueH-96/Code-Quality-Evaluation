class Solution:
    def minElement(self, nums: List[int]) -> int:
        transformed = [sum(int(digit) for digit in str(num)) for num in nums]
        return min(transformed)