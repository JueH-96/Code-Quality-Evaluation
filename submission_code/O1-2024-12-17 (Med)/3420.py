class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Collect all indices where nums[i] == x
        indices = [i for i, val in enumerate(nums) if val == x]
        
        # For each query, we want the query-th occurrence of x
        # which corresponds to indices[query-1] if query <= len(indices)
        # Otherwise, return -1
        result = []
        for query in queries:
            if query <= len(indices):
                result.append(indices[query - 1])
            else:
                result.append(-1)
                
        return result