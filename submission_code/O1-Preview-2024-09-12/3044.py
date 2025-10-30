class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected_elements = set()
        operations = 0
        required_elements = set(range(1, k + 1))
        index = len(nums)
        while collected_elements != required_elements:
            index -= 1
            element = nums[index]
            if 1 <= element <= k:
                collected_elements.add(element)
            operations += 1
        return operations