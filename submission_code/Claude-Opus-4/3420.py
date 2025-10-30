class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where x occurs
        occurrences = []
        for i in range(len(nums)):
            if nums[i] == x:
                occurrences.append(i)
        
        # Process each query
        result = []
        for query in queries:
            # query is 1-based, so we need the (query-1)th index in occurrences
            if query <= len(occurrences):
                result.append(occurrences[query - 1])
            else:
                result.append(-1)
        
        return result