class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 1000000007
        n = len(nums)
        
        # Sort the array to easily get min and max elements
        nums.sort()
        
        # Initialize result
        total = 0
        
        # For each length i from 1 to k
        for length in range(1, k + 1):
            # For each starting position j
            for j in range(n - length + 1):
                # For each possible ending position
                for end in range(j + length - 1, n):
                    # The minimum will be nums[j] (first element in subsequence)
                    # The maximum will be nums[end] (last element in subsequence)
                    total = (total + nums[j] + nums[end]) % MOD
        
        return total