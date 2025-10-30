class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        possible_values = set()
        for num in nums:
            possible_values.add(num)
            possible_values.add(num + 1)

        nums_set = possible_values
        max_length = 0
        num_set = set(nums_set)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_length = 1
                while current_num + 1 in num_set:
                    current_num +=1
                    current_length +=1
                max_length = max(max_length, current_length)
        return max_length