class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        dp = {}
        # dp is a dictionary where keys are alternating sums and values are the maximum product for that sum
        max_p = -1
        
        for num in nums:
            temp = {}
            # Option 1: start a new subsequence with current num (length 1, even index)
            new_sum = num
            new_prod = num
            if new_prod <= limit:
                if new_sum in temp:
                    if new_prod > temp[new_sum]:
                        temp[new_sum] = new_prod
                else:
                    temp[new_sum] = new_prod
            
            # Option 2: add num to existing subsequences in dp
            for old_sum in dp:
                old_prod = dp[old_sum]
                # Adding num as even index (new length is odd)
                new_sum1 = old_sum + num
                new_prod1 = old_prod * num
                if new_prod1 <= limit:
                    if new_sum1 in temp:
                        if new_prod1 > temp[new_sum1]:
                            temp[new_sum1] = new_prod1
                    else:
                        temp[new_sum1] = new_prod1
                # Adding num as odd index (new length is even)
                new_sum2 = old_sum - num
                new_prod2 = old_prod * num
                if new_prod2 <= limit:
                    if new_sum2 in temp:
                        if new_prod2 > temp[new_sum2]:
                            temp[new_sum2] = new_prod2
                    else:
                        temp[new_sum2] = new_prod2
            
            # Merge temp into dp
            for s in temp:
                if s in dp:
                    if temp[s] > dp[s]:
                        dp[s] = temp[s]
                else:
                    dp[s] = temp[s]
        
        if k in dp:
            return dp[k]
        else:
            return -1