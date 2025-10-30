class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Create a list to store the ranges [nums[i] - k, nums[i] + k]
        ranges = []
        for num in nums:
            ranges.append((num - k, num + k))
        
        # Sort the ranges based on the start value
        ranges.sort()
        
        max_beauty = 0
        current_beauty = 0
        # Use a min-heap to keep track of the end values of the current overlapping ranges
        import heapq
        heap = []
        
        for start, end in ranges:
            # Remove ranges from the heap that do not overlap with the current range
            while heap and heap[0] < start:
                heapq.heappop(heap)
                current_beauty -= 1
            # Add the current range's end to the heap
            heapq.heappush(heap, end)
            current_beauty += 1
            # Update the maximum beauty
            if current_beauty > max_beauty:
                max_beauty = current_beauty
        
        return max_beauty