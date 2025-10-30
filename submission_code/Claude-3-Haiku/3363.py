from collections import Counter

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        ans = []
        id_counts = Counter()
        
        for i in range(len(nums)):
            id_counts[nums[i]] += freq[i]
            
            most_frequent = id_counts.most_common(1)
            if most_frequent:
                ans.append(most_frequent[0][0])
            else:
                ans.append(0)
        
        return ans