class Solution:
    def min_operations(self, nums: List[int], k: List[int]) -> int:
        reversed_nums = nums[::-1]
        required = set(range(1, k+1))
        current_elements = set()
        for i in range(len(reversed_nums)):
            current_elements.add(reversed_nums[i])
            if current_elements.issuperset(required):
                return i + 1
        return len(reversed_nums)