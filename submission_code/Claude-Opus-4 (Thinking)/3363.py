class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        import heapq
        from collections import defaultdict
        
        n = len(nums)
        ans = []
        id_count = defaultdict(int)
        max_heap = []  # We'll use negative values for max heap
        
        for i in range(n):
            id_val = nums[i]
            freq_change = freq[i]
            
            # Update the frequency of the ID
            id_count[id_val] += freq_change
            
            # Push the new frequency to the heap if it's positive
            if id_count[id_val] > 0:
                heapq.heappush(max_heap, (-id_count[id_val], id_val))
            
            # Remove stale entries from the top of the heap
            while max_heap and id_count[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            
            # The answer is the maximum frequency
            if max_heap:
                ans.append(-max_heap[0][0])
            else:
                ans.append(0)
        
        return ans