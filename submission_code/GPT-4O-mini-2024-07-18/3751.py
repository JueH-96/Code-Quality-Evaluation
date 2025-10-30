class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        # Count the frequency of each number in nums
        count = Counter(nums)
        
        # Initialize the maximum frequency
        max_freq = 0
        
        # Iterate through all possible values that can be formed by adding x
        for num in range(1, 51):  # Since nums[i] and k are in the range 1 to 50
            # Calculate how many times we can achieve the value `num` by adding x
            if num <= k:
                # We need to find how many elements can be transformed to `num`
                # This means we need to find how many elements can be turned into `num` by adding (num - current_value)
                needed = k - num
                freq = 0
                
                for value in range(1, 51):
                    if value + needed == num:
                        freq += count[value]
                
                max_freq = max(max_freq, freq)
        
        return max_freq