class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        result = []
        
        for index, k in queries:
            # Mark the element at index if not already marked
            if not marked[index]:
                marked[index] = True
            
            # Collect all unmarked elements with their indices
            unmarked_elements = [(nums[i], i) for i in range(n) if not marked[i]]
            
            # Sort unmarked elements by value, then by index
            unmarked_elements.sort()
            
            # Mark the smallest k elements
            for i in range(min(k, len(unmarked_elements))):
                marked[unmarked_elements[i][1]] = True
            
            # Calculate the sum of unmarked elements
            unmarked_sum = sum(nums[i] for i in range(n) if not marked[i])
            result.append(unmarked_sum)
        
        return result