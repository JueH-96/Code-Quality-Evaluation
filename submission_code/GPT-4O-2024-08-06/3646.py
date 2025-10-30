class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Count the frequency of each number in nums
        from collections import Counter
        freq = Counter(nums)
        
        # Initialize the sum of good subsequences
        total_sum = 0
        
        # Iterate over each unique number in nums
        for num in freq:
            # Calculate the contribution of subsequences starting with this number
            # Subsequence of size 1
            count = freq[num]
            sum_contribution = num * count
            
            # Check for subsequences of size 2 and 3
            if num + 1 in freq:
                count_next = freq[num + 1]
                sum_contribution += (num + (num + 1)) * count * count_next
            
            if num + 2 in freq:
                count_next_next = freq[num + 2]
                sum_contribution += (num + (num + 1) + (num + 2)) * count * count_next * count_next_next
            
            # Add the contribution to the total sum
            total_sum = (total_sum + sum_contribution) % MOD
        
        return total_sum