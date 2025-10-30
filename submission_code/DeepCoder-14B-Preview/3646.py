MOD = 10**9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        sum_dp = dict()  # Maps each number to the sum of subsequences ending with it
        count_dp = dict()  # Maps each number to the count of subsequences ending with it
        total = 0
        
        for num in nums:
            temp_sum = 0
            temp_count = 0
            
            # Check if we can extend subsequences ending with num-1
            if (num - 1) in sum_dp:
                temp_sum += sum_dp[num - 1]
                temp_sum += num * count_dp[num - 1]
                temp_count += count_dp[num - 1]
            
            # Check if we can extend subsequences ending with num+1
            if (num + 1) in sum_dp:
                temp_sum += sum_dp[num + 1]
                temp_sum += num * count_dp[num + 1]
                temp_count += count_dp[num + 1]
            
            # Add the single-element subsequence
            temp_sum += num
            temp_count += 1
            
            # Update the dictionaries with the new contributions
            if num in sum_dp:
                sum_dp[num] = (sum_dp[num] + temp_sum) % MOD
                count_dp[num] = (count_dp[num] + temp_count) % MOD
            else:
                sum_dp[num] = temp_sum % MOD
                count_dp[num] = temp_count % MOD
            
            # Add the current contribution to the total sum
            total = (total + temp_sum) % MOD
        
        return total % MOD