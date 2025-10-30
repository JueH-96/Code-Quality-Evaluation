class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total = 0
        # Precompute the powers of 2 modulo MOD
        power_of_two = [1] * (n + 1)
        for i in range(1, n + 1):
            power_of_two[i] = (power_of_two[i-1] * 2) % MOD
        for i in range(n):
            # For each element, it can be the maximum in all subsets that include it and any subset of the previous elements
            # The number of such subsets is 2^i
            # The power contribution is nums[i]^2 * nums[i] * 2^i
            # Wait, no. The power is max^2 * min. If nums[i] is the max, then min is the smallest element in the subset
            # So, for each subset where nums[i] is the max, the min is the smallest element in the subset
            # To handle this, we need to consider all possible subsets where nums[i] is the max and find the sum of min over all such subsets
            # This is complex, so we need a smarter approach
            # Instead, for each element nums[i], we can consider it as the max and find all subsets where it is the max
            # The min in such a subset is the smallest element in the subset
            # The number of subsets where nums[i] is the max and nums[j] is the min is 2^{i-j-1} if j < i, and 1 if j == i
            # So, for each j <= i, the contribution is nums[i]^2 * nums[j] * 2^{i-j-1} if j < i, and nums[i]^2 * nums[j] if j == i
            # So, we can iterate over j from 0 to i and compute this
            # To optimize, we can precompute the sum of nums[j] * 2^{-j-1} for j < i, and then multiply by 2^i
            # But since we are dealing with modulo operations, we need to handle the division carefully
            # Alternatively, we can precompute the sum of nums[j] * 2^{i-j-1} for j < i
            # This can be done by maintaining a running sum
            # Initialize the running sum
            if i == 0:
                running_sum = 0
            else:
                running_sum = (running_sum * 2 + nums[i-1]) % MOD
            # The contribution from j < i is nums[i]^2 * running_sum
            contribution = (nums[i] * nums[i] % MOD) * running_sum % MOD
            # The contribution from j == i is nums[i]^2 * nums[i]
            contribution = (contribution + (nums[i] * nums[i] % MOD) * nums[i] % MOD) % MOD
            total = (total + contribution) % MOD
        return total