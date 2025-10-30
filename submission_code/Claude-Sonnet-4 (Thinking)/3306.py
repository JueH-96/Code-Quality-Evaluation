class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        total_sum = sum(nums)
        
        # Create sorted list of (value, index) pairs
        sorted_pairs = [(nums[i], i) for i in range(n)]
        sorted_pairs.sort()
        
        result = []
        pointer = 0  # Pointer to the current position in sorted_pairs
        
        for index_i, k_i in queries:
            # Mark the element at index_i if not already marked
            if not marked[index_i]:
                marked[index_i] = True
                total_sum -= nums[index_i]
            
            # Mark k_i smallest unmarked elements
            count = 0
            while pointer < n and count < k_i:
                value, idx = sorted_pairs[pointer]
                if not marked[idx]:
                    marked[idx] = True
                    total_sum -= nums[idx]
                    count += 1
                pointer += 1
            
            result.append(total_sum)
        
        return result