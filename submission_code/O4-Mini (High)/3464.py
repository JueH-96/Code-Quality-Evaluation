class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # prefixA = sum of nums[k]*(-1)^k for k < i
        prefixA = 0
        # dp[i] = max total cost for first i elements
        # We only keep current dp, and two best values for transitions
        dp_i = 0
        # best_even = max over j even of (dp[j] - prefixA[j])
        best_even = 0
        # best_odd  = max over j odd  of (dp[j] + prefixA[j])
        # initialize to very small
        best_odd = -10**30
        
        for i, x in enumerate(nums, start=1):
            # update prefixA for index k = i-1
            k = i - 1
            if k & 1:
                # k is odd: (-1)^k = -1
                prefixA -= x
            else:
                # k is even: (-1)^k = +1
                prefixA += x
            
            # dp[i] = max(best_even + prefixA, best_odd - prefixA)
            # best_even and best_odd are based on j < i
            cand1 = best_even + prefixA
            cand2 = best_odd - prefixA
            dp_i = cand1 if cand1 > cand2 else cand2
            
            # update best_even / best_odd for use in future iterations
            if i & 1:
                # i is odd -> j = i goes into best_odd
                # best_odd = max(best_odd, dp[i] + prefixA[i])
                v = dp_i + prefixA
                if v > best_odd:
                    best_odd = v
            else:
                # i is even -> j = i goes into best_even
                # best_even = max(best_even, dp[i] - prefixA[i])
                v = dp_i - prefixA
                if v > best_even:
                    best_even = v
        
        return dp_i