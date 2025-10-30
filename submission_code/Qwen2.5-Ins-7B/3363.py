from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_count = defaultdict(int)
        max_count = 0
        ans = []
        
        for i in range(len(nums)):
            id_count[nums[i]] += freq[i]
            max_count = max(max_count, id_count[nums[i]])
            ans.append(max_count)
        
        return ans