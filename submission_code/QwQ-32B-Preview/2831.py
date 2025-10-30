class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        import math
        counter = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                first_digit = int(str(nums[i])[0])
                last_digit = nums[j] % 10
                if math.gcd(first_digit, last_digit) == 1:
                    counter += 1
        return counter