class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        result = []
        
        # Create array of (value, index) pairs for sorting
        indexed_nums = [(num, i) for i, num in enumerate(nums)]
        indexed_nums.sort()  # Sort by value, then by index
        
        # Keep track of current unmarked sum and next available position in sorted array
        unmarked_sum = sum(nums)
        next_pos = 0
        
        for index, k in queries:
            # Mark element at index if not already marked
            if not marked[index]:
                unmarked_sum -= nums[index]
                marked[index] = True
            
            # Mark k smallest unmarked elements
            count = k
            while count > 0 and next_pos < n:
                val, idx = indexed_nums[next_pos]
                if not marked[idx]:
                    marked[idx] = True
                    unmarked_sum -= val
                    count -= 1
                next_pos += 1
            
            result.append(unmarked_sum)
            
        return result