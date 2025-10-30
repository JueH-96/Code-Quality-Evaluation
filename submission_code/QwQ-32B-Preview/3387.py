class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        import heapq
        n = len(nums)
        m = n // 2
        
        # Sort the array
        nums_sorted = sorted(nums)
        
        # Adjust the first m elements to be <= k
        for i in range(m):
            if nums_sorted[i] > k:
                nums_sorted[i] = k
        
        # Adjust the elements from m to n-1 to be >= k
        for i in range(m, n):
            if nums_sorted[i] < k:
                nums_sorted[i] = k
        
        # Calculate the total operations
        operations = 0
        for original, new in zip(nums, nums_sorted):
            operations += abs(original - new)
        
        return operations