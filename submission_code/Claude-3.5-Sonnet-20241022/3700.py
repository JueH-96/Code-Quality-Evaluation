class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        result = 0
        
        # Try each possible middle element position
        for i in range(2, n-2):
            # For each middle element
            middle = nums[i]
            
            # Count elements before middle position
            before = nums[:i]
            # Count elements after middle position
            after = nums[i+1:]
            
            # For each possible combination of 2 elements before middle
            for j in range(len(before)):
                for k in range(j+1, len(before)):
                    left = [before[j], before[k]]
                    
                    # For each possible combination of 2 elements after middle
                    for m in range(len(after)):
                        for n in range(m+1, len(after)):
                            right = [after[m], after[n]]
                            
                            # Create the subsequence
                            seq = left + [middle] + right
                            
                            # Count frequencies
                            freq = {}
                            for num in seq:
                                freq[num] = freq.get(num, 0) + 1
                            
                            # Find maximum frequency
                            max_freq = max(freq.values())
                            
                            # Check if middle element has unique maximum frequency
                            if freq[middle] == max_freq and list(freq.values()).count(max_freq) == 1:
                                result = (result + 1) % MOD
        
        return result