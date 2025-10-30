class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        import heapq
        
        count = defaultdict(int)
        max_heap = []
        ans = []
        max_freq = 0
        
        for i in range(len(nums)):
            id = nums[i]
            change = freq[i]
            count[id] += change
            
            if count[id] > max_freq:
                max_freq = count[id]
            
            if count[id] == 0:
                del count[id]
            
            if max_freq == 0:
                ans.append(0)
            else:
                # Find the current maximum frequency in the count dictionary
                current_max = max(count.values())
                ans.append(current_max)
        
        return ans