class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Function to check if nums can be converted to zero array with given queries
        def can_convert_to_zero(nums, queries):
            temp_nums = nums[:]
            for l, r in queries:
                for i in range(l, r + 1):
                    if temp_nums[i] > 0:
                        temp_nums[i] -= 1
            return all(x == 0 for x in temp_nums)
        
        # Try to remove each query and check if we can still convert nums to zero array
        removable_count = 0
        for i in range(m):
            remaining_queries = queries[:i] + queries[i+1:]
            if can_convert_to_zero(nums, remaining_queries):
                removable_count += 1
        
        # If no queries can be removed and still convert to zero array, return -1
        if removable_count == 0 and not can_convert_to_zero(nums, queries):
            return -1
        
        return removable_count