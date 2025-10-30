class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a list to store the indices of the element x in nums
        indices = [i for i, num in enumerate(nums) if num == x]

        # Initialize the result list with -1 for all queries
        result = [-1] * len(queries)

        # Iterate through the queries
        for i, query in enumerate(queries):
            # Check if the query is within the range of the number of occurrences of x
            if query <= len(indices):
                # The index of the query-th occurrence of x is indices[query-1]
                result[i] = indices[query - 1]

        return result