class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where x occurs
        x_indices = []
        for i, num in enumerate(nums):
            if num == x:
                x_indices.append(i)
        
        # Process each query
        result = []
        for query in queries:
            if query <= len(x_indices):
                # query-th occurrence exists (1-indexed), so get index at query-1
                result.append(x_indices[query - 1])
            else:
                result.append(-1)
        
        return result