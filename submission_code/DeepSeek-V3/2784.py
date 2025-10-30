class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total = 0
        prefix = 0
        for i in range(n):
            current = nums[i]
            # Calculate the contribution of current as the maximum
            # The number of subsets where current is the maximum is 2^i
            # But since we need to include at least one element, it's 2^i - 1
            # However, for the current element, it's always included as the maximum
            # So for the current element, the number of subsets where it is the maximum is 2^i
            # But for the minimum, it's the current element itself
            # So the power is current^2 * current * (2^i)
            # Wait, no. For the current element as the maximum, the minimum can be any element in the subset
            # So for each subset where current is the maximum, the minimum is the smallest element in the subset
            # To handle this, we need to consider the prefix sum of the elements before current
            # The power for a subset where current is the maximum and some element x is the minimum is current^2 * x
            # The number of such subsets is 2^{i - j - 1}, where j is the index of x
            # To optimize, we can precompute the prefix sum of the elements before current
            # For each current, the total power is current^2 * (sum_{j=0}^{i-1} nums[j] * 2^{i-j-1})
            # Which can be rewritten as current^2 * (sum_{j=0}^{i-1} nums[j] * 2^{i-1} / 2^j)
            # Or current^2 * 2^{i-1} * sum_{j=0}^{i-1} nums[j] / 2^j
            # To compute this efficiently, we can maintain a running sum of nums[j] / 2^j
            # So, prefix_sum = sum_{j=0}^{i-1} nums[j] / 2^j
            # Then, the contribution is current^2 * 2^{i-1} * prefix_sum
            # But since we are dealing with integers, we need to handle the division carefully
            # Alternatively, we can precompute the prefix sum of nums[j] * 2^{i-1 - j}
            # Which is equivalent to 2^{i-1} * sum_{j=0}^{i-1} nums[j] / 2^j
            # So, we can compute prefix_sum as sum_{j=0}^{i-1} nums[j] * 2^{i-1 - j}
            # To compute this efficiently, we can maintain a running sum where each step multiplies by 2 and adds nums[j]
            # So, prefix = prefix * 2 + nums[j]
            # Wait, no. For each j, the term is nums[j] * 2^{i-1 - j}
            # So, for i increasing, the exponent decreases
            # To compute this, we can maintain a prefix sum where each step multiplies by 2 and adds nums[j]
            # So, prefix = prefix * 2 + nums[j]
            # Then, for the current i, the prefix sum is prefix
            # So, the contribution is current^2 * prefix
            # But we need to adjust the prefix for the current i
            # So, for each i, the prefix is the sum of nums[j] * 2^{i-1 - j} for j < i
            # Which is equivalent to prefix = prefix * 2 + nums[i-1]
            # So, we can compute the prefix as follows:
            if i > 0:
                prefix = (prefix * 2 + nums[i-1]) % MOD
            else:
                prefix = 0
            # Now, the contribution is current^2 * prefix
            contribution = (current * current % MOD) * prefix % MOD
            # Also, the current element itself forms a subset where it is both the maximum and minimum
            # So, the power is current^2 * current
            contribution += (current * current % MOD) * current % MOD
            total = (total + contribution) % MOD
        return total