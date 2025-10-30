class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        result = []
        
        for index, k in queries:
            # Mark the element at the given index
            if not marked[index]:
                marked[index] = True
            
            # Find k smallest unmarked elements and mark them
            unmarked_indices = [i for i in range(n) if not marked[i]]
            unmarked_indices.sort(key=lambda x: (nums[x], x))
            
            for i in range(min(k, len(unmarked_indices))):
                marked[unmarked_indices[i]] = True
            
            # Calculate the sum of unmarked elements
            unmarked_sum = sum(nums[i] for i in range(n) if not marked[i])
            result.append(unmarked_sum)
        
        return result