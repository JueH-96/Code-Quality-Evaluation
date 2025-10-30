class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = set()
        result = []
        total_sum = sum(nums)
        
        # Create sorted list of (value, index) pairs
        # This ensures we get smallest values first, and smallest indices for ties
        sorted_pairs = sorted((nums[i], i) for i in range(n))
        
        for index, k in queries:
            # Mark the element at index if not already marked
            if index not in marked:
                marked.add(index)
                total_sum -= nums[index]
            
            # Find and mark k smallest unmarked elements
            count = 0
            for value, idx in sorted_pairs:
                if count >= k:
                    break
                if idx not in marked:
                    marked.add(idx)
                    total_sum -= value
                    count += 1
            
            result.append(total_sum)
        
        return result