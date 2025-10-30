class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = {}
        ans = []
        for i in range(len(nums)):
            id_val = nums[i]
            freq_val = freq[i]
            
            if id_val in counts:
                counts[id_val] += freq_val
            else:
                counts[id_val] = freq_val
            
            max_freq = 0
            if len(counts) == 0:
                ans.append(0)
            else:
                for id_val in counts:
                    max_freq = max(max_freq, counts[id_val])
                ans.append(max_freq)
        return ans