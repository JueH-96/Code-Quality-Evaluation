class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if num < 10)
        double_digit_sum = sum(num for num in nums if num >= 10)
        total_sum = sum(nums)
        
        # Alice wins if the sum of her numbers is strictly greater than Bob's numbers
        # Bob's sum would be total_sum - Alice's sum
        # So, Alice wins if single_digit_sum > total_sum - single_digit_sum
        # or if double_digit_sum > total_sum - double_digit_sum
        return single_digit_sum > total_sum - single_digit_sum or double_digit_sum > total_sum - double_digit_sum