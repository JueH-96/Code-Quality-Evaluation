class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        import heapq

        id_count = defaultdict(int)
        max_heap = []
        ans = []

        for i in range(len(nums)):
            id_count[nums[i]] += freq[i]
            
            if id_count[nums[i]] > 0:
                heapq.heappush(max_heap, (-id_count[nums[i]], nums[i]))
            
            while max_heap and -max_heap[0][0] > id_count[max_heap[0][1]]:
                heapq.heappop(max_heap)
            
            if max_heap:
                ans.append(-max_heap[0][0])
            else:
                ans.append(0)

        return ans