class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        if k == 0:
            return reduce(or_, nums)
        
        # If k is large enough to potentially reduce the array to one element
        if k >= len(nums) - 1:
            # Compute the minimum possible OR by ANDing all elements
            result = nums[0]
            for num in nums[1:]:
                result &= num
            return result
        
        # Use a priority queue to greedily combine the smallest resulting AND values
        import heapq
        heap = []
        for i in range(len(nums) - 1):
            combined_value = nums[i] & nums[i + 1]
            heapq.heappush(heap, (combined_value, i, i + 1))
        
        # Union-Find structure to manage merging elements
        parent = list(range(len(nums)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        # Perform up to k operations
        operations = 0
        while operations < k and heap:
            val, i, j = heapq.heappop(heap)
            # Ensure that the indices are still valid and represent current groups
            if find(i) != find(j):
                union(i, j)
                operations += 1
                # Update the nums array to reflect the AND operation
                nums[i] = val
                # Push new possible combinations into the heap
                if i > 0 and find(i - 1) != find(i):
                    new_val = nums[find(i - 1)] & nums[find(i)]
                    heapq.heappush(heap, (new_val, find(i - 1), find(i)))
                if j + 1 < len(nums) and find(j + 1) != find(j):
                    new_val = nums[find(j)] & nums[find(j + 1)]
                    heapq.heappush(heap, (new_val, find(j), find(j + 1)))
        
        # Calculate the OR of the final array
        final_or = 0
        seen = set()
        for i in range(len(nums)):
            root = find(i)
            if root not in seen:
                final_or |= nums[root]
                seen.add(root)
        
        return final_or