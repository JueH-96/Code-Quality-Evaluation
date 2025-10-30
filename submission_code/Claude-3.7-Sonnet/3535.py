class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # DP[i][j] = number of ways to construct a valid monotonic pair for the
        # first i+1 elements of nums, where arr1[i] = j
        DP = []
        for i in range(n):
            DP.append([0] * (nums[i] + 1))
        
        # Base case: There's only one way to set the first element of arr1
        for j in range(nums[0] + 1):
            DP[0][j] = 1
        
        # Fill the DP table
        for i in range(1, n):
            # Use prefix sum for efficient calculation
            prefix_sum = [0]
            for k in range(nums[i-1] + 1):
                prefix_sum.append(prefix_sum[-1] + DP[i-1][k])
            
            for j in range(nums[i] + 1):
                # To maintain monotonicity:
                # 1. arr1[i-1] <= arr1[i] (i.e., k <= j)
                # 2. arr2[i-1] >= arr2[i] (i.e., nums[i-1] - k >= nums[i] - j)
                #    which gives k <= j + (nums[i-1] - nums[i])
                upper_bound = min(j, nums[i-1], j + (nums[i-1] - nums[i]))
                
                if upper_bound < 0:
                    DP[i][j] = 0
                else:
                    # Count valid ways
                    DP[i][j] = prefix_sum[upper_bound + 1] % MOD
        
        # Sum up all valid configurations
        total = 0
        for j in range(nums[n-1] + 1):
            total = (total + DP[n-1][j]) % MOD
            
        return total