class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        import heapq
        from collections import defaultdict
        
        n = len(nums)
        ans = [0] * n
        
        # Dictionary to track the current frequency of each ID
        id_freq = defaultdict(int)
        
        # Max heap to efficiently find the most frequent ID
        # Using negative frequencies for max heap behavior
        max_heap = []
        
        for i in range(n):
            id_num = nums[i]
            change = freq[i]
            
            # Update the frequency of the current ID
            id_freq[id_num] += change
            
            # Push the updated frequency to the heap
            heapq.heappush(max_heap, (-id_freq[id_num], id_num))
            
            # Remove outdated entries from the heap
            while max_heap and -max_heap[0][0] != id_freq[max_heap[0][1]]:
                heapq.heappop(max_heap)
            
            # The most frequent ID count is at the top of the heap
            ans[i] = -max_heap[0][0] if max_heap else 0
        
        return ans