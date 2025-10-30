from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        ans = []
        max_freq = 0
        max_id = None
        
        for i in range(len(nums)):
            id_val = nums[i]
            freq_change = freq[i]
            
            count[id_val] += freq_change
            
            if count[id_val] > max_freq:
                max_freq = count[id_val]
                max_id = id_val
            elif count[id_val] == max_freq and max_id != id_val:
                max_id = id_val
            
            if max_freq == 0:
                ans.append(0)
            else:
                ans.append(max_freq)
        
        return ans