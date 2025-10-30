class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where x occurs
        x_indices = []
        for i in range(len(nums)):
            if nums[i] == x:
                x_indices.append(i)
                
        # Process each query
        result = []
        for q in queries:
            # If query asks for occurrence beyond what exists
            if q > len(x_indices):
                result.append(-1)
            else:
                # q-1 since queries are 1-indexed but array is 0-indexed
                result.append(x_indices[q-1])
                
        return result