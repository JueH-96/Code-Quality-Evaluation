class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        total_sum = sum(nums)
        
        # Create a list of (value, index) pairs and sort them
        # This will help us find smallest unmarked elements efficiently
        value_index_pairs = [(nums[i], i) for i in range(n)]
        value_index_pairs.sort()
        
        # Keep track of which elements in our sorted list we've already processed
        processed_idx = 0
        
        result = []
        
        for index_i, k_i in queries:
            # Mark the element at index_i if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                total_sum -= nums[index_i]
            
            # Mark k_i smallest unmarked elements
            marked_count = 0
            temp_idx = processed_idx
            
            # Find and mark k_i smallest unmarked elements
            while marked_count < k_i and temp_idx < n:
                value, idx = value_index_pairs[temp_idx]
                if not marked[idx]:
                    marked[idx] = True
                    total_sum -= nums[idx]
                    marked_count += 1
                temp_idx += 1
            
            # Update processed_idx to skip over elements we've already marked
            while processed_idx < n and marked[value_index_pairs[processed_idx][1]]:
                processed_idx += 1
            
            result.append(total_sum)
        
        return result