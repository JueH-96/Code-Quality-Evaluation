class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        from collections import defaultdict
        import heapq
        
        n = len(nums)
        
        # Function to calculate the cost to make all elements in a subarray equal to the median
        def cost_to_make_equal(subarray):
            subarray.sort()
            median = subarray[len(subarray) // 2]
            return sum(abs(num - median) for num in subarray)
        
        # Priority queue to store the minimum costs of forming subarrays of size x
        min_heap = []
        
        # Calculate the cost for all possible subarrays of size x
        for i in range(n - x + 1):
            sub_cost = cost_to_make_equal(nums[i:i+x])
            heapq.heappush(min_heap, sub_cost)
        
        # Sum up the k smallest costs from the heap
        total_min_operations = 0
        for _ in range(k):
            total_min_operations += heapq.heappop(min_heap)
        
        return total_min_operations