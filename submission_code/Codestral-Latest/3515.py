class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if num < 10)
        double_digit_sum = sum(num for num in nums if num >= 10)

        total_sum = sum(nums)

        # Alice can win if the sum of her chosen numbers is strictly greater than the sum of Bob's numbers
        return single_digit_sum > (total_sum - single_digit_sum) or double_digit_sum > (total_sum - double_digit_sum)