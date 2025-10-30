class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        from collections import Counter
        
        MOD = 10**9 + 7
        
        # Count the frequency of each number in nums
        count = Counter(nums)
        
        # To store the total sum of good subsequences
        total_sum = 0
        
        # We need to consider each number and its neighbors
        for num in range(100001):
            if num in count:
                # Calculate the contribution of subsequences starting with num
                current_count = count[num]
                
                # Contribution from single elements
                total_sum += current_count * num % MOD
                total_sum %= MOD
                
                # Contribution from pairs (num, num + 1)
                if num + 1 in count:
                    total_sum += current_count * count[num + 1] * (num + 1) % MOD
                    total_sum %= MOD
                
                # Contribution from pairs (num, num - 1)
                if num - 1 in count:
                    total_sum += current_count * count[num - 1] * (num - 1) % MOD
                    total_sum %= MOD
        
        return total_sum