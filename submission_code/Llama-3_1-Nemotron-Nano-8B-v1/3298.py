from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        max_length = 0
        for num in nums:
            option1 = dp.get(num - 1, 0) + 1
            option2 = dp.get(num, 0) + 1
            if option1 > option2:
                new_value = num
                new_length = option1
            else:
                new_value = num + 1
                new_length = option2
            if new_value in dp:
                if new_length > dp[new_value]:
                    dp[new_value] = new_length
            else:
                dp[new_value] = new_length
            if new_length > max_length:
                max_length = new_length
        return max_length