from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 5:
            return 0
        
        # Count the frequency of each element in the array
        freq = Counter(nums)
        
        # Find the number of subsequences with a unique middle mode
        count = 0
        for i in range(n-4):
            for j in range(i+1, n-3):
                for k in range(j+1, n-2):
                    middle = nums[k]
                    if freq[middle] == max(freq.values()) and all(freq[nums[l]] < freq[middle] for l in [i, i+1, j, j+1, k+1, k+2]):
                        count += 1
        
        return count % (10**9 + 7)