class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        sum_ending_with = {}
        count_ending_with = {}
        
        for num in nums:
            # Calculate new contributions
            new_sum = num  # Single-element subsequence
            new_count = 1
            
            # Extend subsequences ending with num-1
            if num-1 in sum_ending_with:
                new_sum = (new_sum + sum_ending_with[num-1] + count_ending_with[num-1] * num) % MOD
                new_count = (new_count + count_ending_with[num-1]) % MOD
            
            # Extend subsequences ending with num+1
            if num+1 in sum_ending_with:
                new_sum = (new_sum + sum_ending_with[num+1] + count_ending_with[num+1] * num) % MOD
                new_count = (new_count + count_ending_with[num+1]) % MOD
            
            # Add to existing values
            sum_ending_with[num] = sum_ending_with.get(num, 0) + new_sum
            count_ending_with[num] = count_ending_with.get(num, 0) + new_count
            sum_ending_with[num] %= MOD
            count_ending_with[num] %= MOD
        
        # Sum all subsequences
        total = 0
        for s in sum_ending_with.values():
            total = (total + s) % MOD
        return total