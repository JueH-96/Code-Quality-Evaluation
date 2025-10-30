class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        def check(length):
            # Try all possible subsequences of given length
            dp = [[float('-inf')] * (k + 1) for _ in range(length)]
            dp[0][0] = nums[0]
            
            # For first position, we can take any number
            for i in range(n):
                dp[0][0] = nums[i]
                
                # For each position and number of differences allowed
                for pos in range(1, length):
                    for diff in range(k + 1):
                        # Try all possible previous numbers
                        for prev in range(i):
                            # If current number is different from nums[prev]
                            # we need one more difference
                            if nums[i] != nums[prev] and diff > 0:
                                dp[pos][diff] = max(dp[pos][diff], 
                                                  dp[pos-1][diff-1])
                            # If current number is same as nums[prev]
                            if nums[i] == nums[prev]:
                                dp[pos][diff] = max(dp[pos][diff], 
                                                  dp[pos-1][diff])
                                
                if dp[length-1][k] != float('-inf'):
                    return True
            return False
        
        # Binary search on the length
        left, right = 1, n
        ans = 1
        
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return ans