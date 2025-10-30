from collections import defaultdict

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        n = len(nums)
        count = 0
        
        for i in range(n):
            freq = defaultdict(int)
            current_distinct = 0
            min_j = -1
            
            for j in range(i, n):
                num = nums[j]
                if freq[num] == 0:
                    current_distinct += 1
                freq[num] += 1
                
                if current_distinct == total_distinct:
                    min_j = j
                    break
            
            if min_j != -1:
                count += (n - min_j)
        
        return count