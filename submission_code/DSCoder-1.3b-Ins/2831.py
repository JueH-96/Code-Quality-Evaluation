class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def gcd(x, y):
            while(y):
                x, y = y, x % y
            return x

        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                first_digit = nums[i] % 10
                second_digit = nums[j] % 10
                if gcd(first_digit, second_digit) == 1:
                    count += 1
        return count