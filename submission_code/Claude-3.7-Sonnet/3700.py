from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        count = 0
        
        # Iterate through all possible middle positions
        for i in range(2, n - 2):
            middle_val = nums[i]
            
            # Iterate through all pairs of positions to the left of the middle
            for j in range(i):
                for k in range(j+1, i):
                    # Iterate through all pairs of positions to the right of the middle
                    for l in range(i+1, n):
                        for m in range(l+1, n):
                            # Subsequence is [nums[j], nums[k], nums[i], nums[l], nums[m]]
                            seq = [nums[j], nums[k], middle_val, nums[l], nums[m]]
                            
                            # Check if middle element is a unique mode
                            freq = Counter(seq)
                            middle_freq = freq[middle_val]
                            is_unique_mode = True
                            
                            for val, f in freq.items():
                                if val != middle_val and f >= middle_freq:
                                    is_unique_mode = False
                                    break
                            
                            if is_unique_mode:
                                count = (count + 1) % MOD
        
        return count