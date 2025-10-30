class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        E = 0  # count of even numbers
        O = 0  # count of odd numbers
        dp0 = 0  # max length ending with even
        dp1 = 0  # max length ending with odd
        
        for num in nums:
            parity = num % 2
            if parity == 0:
                E += 1
                new_dp0 = max(dp0, dp1 + 1)
                new_dp1 = dp1
            else:
                O += 1
                new_dp1 = max(dp1, dp0 + 1)
                new_dp0 = dp0
            dp0, dp1 = new_dp0, new_dp1
        
        same_max = max(E, O)
        alt_max = max(dp0, dp1)
        return max(same_max, alt_max)