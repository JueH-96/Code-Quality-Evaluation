class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Dictionary to store the indices of occurrences of x
        occurrence_indices = []
        
        # Collect all indices where element x occurs in nums
        for index, value in enumerate(nums):
            if value == x:
                occurrence_indices.append(index)
        
        # Result list to store the answers for each query
        result = []
        
        # Process each query
        for query in queries:
            # Check if the occurrence count asked in the query is available
            if query <= len(occurrence_indices):
                result.append(occurrence_indices[query - 1])
            else:
                result.append(-1)
        
        return result