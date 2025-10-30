class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from collections import Counter
        import heapq
        
        # Count the frequency of each power of 2 in nums
        count = Counter(nums)
        
        # Priority queue to process the largest elements first
        max_heap = []
        for num, freq in count.items():
            heapq.heappush(max_heap, (-num, freq))
        
        operations = 0
        while max_heap:
            # Get the largest element
            num, freq = heapq.heappop(max_heap)
            num = -num
            
            # If the current largest number is less than or equal to target, use it
            if num <= target:
                # Calculate how many times we can use this number
                max_use = min(freq, target // num)
                target -= max_use * num
                freq -= max_use
            
            # If target is met
            if target == 0:
                return operations
            
            # If the number can still be split and used
            if num > 1 and freq > 0:
                # Split the number and add back to the heap
                half_num = num // 2
                heapq.heappush(max_heap, (-half_num, 2 * freq))
                operations += 1
        
        # If we exit the loop without meeting the target
        return -1