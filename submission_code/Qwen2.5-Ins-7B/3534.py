from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(num1, num2):
            str_num1, str_num2 = str(num1), str(num2)
            if len(str_num1) != len(str_num2):
                return False
            diff_count = 0
            for i in range(len(str_num1)):
                if str_num1[i] != str_num2[i]:
                    diff_count += 1
                    if diff_count > 2:
                        return False
            return diff_count == 1 or (diff_count == 2 and str_num1[0] == str_num2[0])
        
        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        return count