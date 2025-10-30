class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        result = 0
        
        for i in range(2, n - 2):  # middle position
            mid_val = nums[i]
            
            # Choose 2 positions before i
            for a in range(i):
                for b in range(a + 1, i):
                    # Choose 2 positions after i
                    for c in range(i + 1, n):
                        for d in range(c + 1, n):
                            # Form the subsequence values
                            values = [nums[a], nums[b], mid_val, nums[c], nums[d]]
                            
                            # Count frequencies
                            freq = {}
                            for val in values:
                                freq[val] = freq.get(val, 0) + 1
                            
                            # Check if mid_val is the unique mode
                            max_freq = max(freq.values())
                            modes = [val for val, f in freq.items() if f == max_freq]
                            
                            if len(modes) == 1 and modes[0] == mid_val:
                                result = (result + 1) % MOD
        
        return result