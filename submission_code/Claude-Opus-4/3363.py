class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        import heapq
        
        # Dictionary to store the current count of each ID
        id_count = defaultdict(int)
        
        # Max heap to track frequencies (negative values for max heap)
        # Each element is (-frequency, id)
        max_heap = []
        
        # Result array
        ans = []
        
        n = len(nums)
        
        for i in range(n):
            id_val = nums[i]
            freq_change = freq[i]
            
            # Update the count for this ID
            id_count[id_val] += freq_change
            
            # Push the new frequency to the heap
            if id_count[id_val] > 0:
                heapq.heappush(max_heap, (-id_count[id_val], id_val))
            
            # Clean up the heap - remove outdated entries
            while max_heap and id_count[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            
            # Get the maximum frequency
            if max_heap:
                ans.append(-max_heap[0][0])
            else:
                ans.append(0)
        
        return ans