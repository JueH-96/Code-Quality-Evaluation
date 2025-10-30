from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            first_digit = int(str(nums[i])[0])
            for j in range(i + 1, len(nums)):
                last_digit = nums[j] % 10
                if gcd(first_digit, last_digit) == 1:
                    count += 1
        return count