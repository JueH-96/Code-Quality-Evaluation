class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        result = 0
        
        # Generate all possible subsequences of length 5
        from itertools import combinations
        
        for indices in combinations(range(n), 5):
            subseq = [nums[i] for i in indices]
            
            # Check if middle element is unique mode
            middle = subseq[2]
            freq = {}
            for num in subseq:
                freq[num] = freq.get(num, 0) + 1
            
            max_freq = max(freq.values())
            modes = [num for num, f in freq.items() if f == max_freq]
            
            # Check if middle is the unique mode
            if len(modes) == 1 and modes[0] == middle:
                result = (result + 1) % MOD
        
        return result