class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7

        # Initialize a dictionary to store the count of each number
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1

        # Initialize the total sum
        total_sum = 0

        # Iterate through the dictionary to calculate the sum of good subsequences
        for num in count:
            total_sum = (total_sum + count[num] * (count[num] + 1) // 2) % MOD

        return total_sum