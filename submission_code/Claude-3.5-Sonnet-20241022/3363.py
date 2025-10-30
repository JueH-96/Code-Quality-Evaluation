class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        from sortedcontainers import SortedList
        
        # Track frequency of each ID
        id_freq = defaultdict(int)
        # Track frequencies in sorted order
        freq_list = SortedList()
        ans = []
        
        for i in range(len(nums)):
            old_freq = id_freq[nums[i]]
            if old_freq > 0:
                freq_list.remove(old_freq)
                
            id_freq[nums[i]] += freq[i]
            new_freq = id_freq[nums[i]]
            
            if new_freq > 0:
                freq_list.add(new_freq)
                
            ans.append(freq_list[-1] if freq_list else 0)
            
        return ans