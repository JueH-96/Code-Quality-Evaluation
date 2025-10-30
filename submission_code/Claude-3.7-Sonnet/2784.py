class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total_power = 0
        
        # Precompute powers of 2 to avoid recalculation
        powers_of_2 = [1]
        for k in range(1, n):
            powers_of_2.append((powers_of_2[-1] * 2) % MOD)
        
        for i in range(n):
            for j in range(i, n):
                min_val = nums[i]
                max_val = nums[j]
                
                # For a given min-max pair (nums[i], nums[j]),
                # calculate how many subsets have this pair
                if i == j:
                    # Just the single element itself
                    subset_count = 1
                else:
                    # Elements between i and j can be either included or excluded
                    subset_count = powers_of_2[j-i-1]
                
                # Calculate power for this min-max pair
                power = (max_val**2 % MOD) * min_val % MOD
                contribution = (power * subset_count) % MOD
                total_power = (total_power + contribution) % MOD
                
        return total_power