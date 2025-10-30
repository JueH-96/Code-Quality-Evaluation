class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number
        freq = Counter(nums)
        
        # We need to find the maximum length of a valid sequence
        max_length = 0
        
        # Sort the numbers to consider them in increasing order
        unique_nums = sorted(freq.keys())
        
        # Dictionary to store the maximum length of a sequence ending with a specific number
        dp = {}
        
        # Iterate over each unique number
        for num in unique_nums:
            # Start a new sequence with the current number
            dp[num] = freq[num]
            
            # Check if it's possible to extend a sequence to this number
            base = 1
            while base <= num:
                if num % base == 0:
                    prev_num = num // base
                    if prev_num in dp and prev_num != num:
                        dp[num] = max(dp[num], dp[prev_num] + freq[num])
                base *= 2
            
            # Update the maximum length found so far
            max_length = max(max_length, dp[num])
        
        return max_length