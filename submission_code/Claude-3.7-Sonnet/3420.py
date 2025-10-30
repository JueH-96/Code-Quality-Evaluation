class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where x appears in nums
        x_indices = [i for i, num in enumerate(nums) if num == x]
        
        # Process each query
        result = []
        for query in queries:
            # Check if the requested occurrence exists
            if query <= len(x_indices):
                # Return the index of the query-th occurrence
                # Need to adjust for 1-indexed queries
                result.append(x_indices[query - 1])
            else:
                # Not enough occurrences of x in nums
                result.append(-1)
        
        return result