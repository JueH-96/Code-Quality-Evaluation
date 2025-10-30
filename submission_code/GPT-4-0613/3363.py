from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_count = defaultdict(int)
        count_ids = defaultdict(set)
        max_count = 0
        ans = []
        
        for i in range(len(nums)):
            if id_count[nums[i]] > 0:
                count_ids[id_count[nums[i]]].remove(nums[i])
                if len(count_ids[id_count[nums[i]]]) == 0:
                    del count_ids[id_count[nums[i]]]
            
            id_count[nums[i]] += freq[i]
            
            if id_count[nums[i]] > 0:
                count_ids[id_count[nums[i]]].add(nums[i])
            
            if freq[i] > 0:
                max_count = max(max_count, id_count[nums[i]])
            
            if max_count > 0 and len(count_ids[max_count]) == 0:
                max_count -= 1
            
            ans.append(max_count)
        
        return ans