class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where x occurs
        occurrences = []
        for i, num in enumerate(nums):
            if num == x:
                occurrences.append(i)
        
        # Process each query
        result = []
        for query in queries:
            if query <= len(occurrences):
                # query-th occurrence exists (1-indexed, so we use query-1 for 0-indexed array)
                result.append(occurrences[query - 1])
            else:
                # Not enough occurrences
                result.append(-1)
        
        return result