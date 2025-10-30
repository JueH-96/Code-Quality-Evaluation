class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = set()
        result = []
        
        # Create a list of (value, index) pairs and sort them
        # This helps us find smallest unmarked elements efficiently
        value_index_pairs = [(nums[i], i) for i in range(n)]
        value_index_pairs.sort()
        
        # Calculate initial sum
        total_sum = sum(nums)
        
        # Pointer to track position in sorted array
        pointer = 0
        
        for index, k in queries:
            # Mark the element at index if not already marked
            if index not in marked:
                marked.add(index)
                total_sum -= nums[index]
            
            # Mark k smallest unmarked elements
            count = 0
            temp_pointer = pointer
            
            # First pass: find k smallest unmarked elements starting from pointer
            while temp_pointer < n and count < k:
                val, idx = value_index_pairs[temp_pointer]
                if idx not in marked:
                    marked.add(idx)
                    total_sum -= val
                    count += 1
                temp_pointer += 1
            
            # Update pointer to skip already processed elements
            while pointer < n and value_index_pairs[pointer][1] in marked:
                pointer += 1
            
            result.append(total_sum)
        
        return result