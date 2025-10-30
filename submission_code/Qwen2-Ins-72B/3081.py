class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        counter = Counter(nums)
        maxHeap = [-cnt for cnt in counter.values()]
        heapify(maxHeap)
        
        while len(maxHeap) > 1:
            first = heappop(maxHeap)
            second = heappop(maxHeap)
            
            if first + 1 < 0:
                heappush(maxHeap, first + 1)
            if second + 1 < 0:
                heappush(maxHeap, second + 1)
        
        return -maxHeap[0] if maxHeap else 0