import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                num1 = nums[i]
                num2 = nums[j]

                first_digit_num1 = 0
                temp_num1 = num1
                while temp_num1 > 0:
                    first_digit_num1 = temp_num1
                    temp_num1 //= 10
                if num1 == 0:
                    first_digit_num1 = 0
                elif num1 < 10:
                    first_digit_num1 = num1


                last_digit_num2 = num2 % 10

                if math.gcd(first_digit_num1, last_digit_num2) == 1:
                    count += 1
        return count