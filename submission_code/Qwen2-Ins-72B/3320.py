class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        first_element = nums[0]
        second_element = nums[1]
        initial_score = first_element + second_element
        operation_count = 1
        index = 2
        
        while index < len(nums) - 1:
            if nums[index] + nums[index + 1] == initial_score:
                operation_count += 1
                index += 2
            else:
                break
                
        return operation_count