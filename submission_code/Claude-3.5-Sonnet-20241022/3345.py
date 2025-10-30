class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] represents number of subsequences using first i elements that sum to j
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(n):
            for j in range(k + 1):
                # Don't take current element
                dp[i + 1][j] = dp[i][j]
                
                # Take current element if possible
                if j >= nums[i]:
                    dp[i + 1][j] = (dp[i + 1][j] + dp[i][j - nums[i]]) % MOD
        
        result = 0
        
        # For each subsequence, count how many subsequences it has with sum k
        def countSubsequences(index, curr_sum, subset):
            if curr_sum == k:
                return 1
            if curr_sum > k or index >= n:
                return 0
                
            count = 0
            # Skip current element
            if index + 1 not in subset:
                count = (count + countSubsequences(index + 1, curr_sum, subset)) % MOD
            
            # Take current element if it's in our subsequence
            if index + 1 in subset:
                count = (count + countSubsequences(index + 1, curr_sum + nums[index], subset)) % MOD
                
            return count
        
        # Generate all subsequences
        for mask in range(1, 1 << n):
            subset = set()
            curr_sum = 0
            for i in range(n):
                if mask & (1 << i):
                    subset.add(i + 1)
                    curr_sum += nums[i]
            
            if curr_sum >= k:
                power = countSubsequences(0, 0, subset)
                result = (result + power) % MOD
                
        return result