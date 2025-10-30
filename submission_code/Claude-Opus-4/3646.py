class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # count[x] = number of good subsequences ending with value x
        # sum_val[x] = sum of all good subsequences ending with value x
        count = {}
        sum_val = {}
        
        for num in nums:
            new_count = 1  # subsequence containing just this number
            new_sum = num  # sum of subsequence containing just this number
            
            # Check if we can extend subsequences ending with num-1
            if num - 1 in count:
                new_count = (new_count + count[num - 1]) % MOD
                new_sum = (new_sum + sum_val[num - 1] + count[num - 1] * num) % MOD
            
            # Check if we can extend subsequences ending with num+1
            if num + 1 in count:
                new_count = (new_count + count[num + 1]) % MOD
                new_sum = (new_sum + sum_val[num + 1] + count[num + 1] * num) % MOD
            
            # Update the values for subsequences ending with num
            if num in count:
                count[num] = (count[num] + new_count) % MOD
                sum_val[num] = (sum_val[num] + new_sum) % MOD
            else:
                count[num] = new_count
                sum_val[num] = new_sum
        
        # Sum all the subsequence sums
        result = 0
        for s in sum_val.values():
            result = (result + s) % MOD
        
        return result