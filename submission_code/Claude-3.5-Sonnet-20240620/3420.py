class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Create a dictionary to store the indices of x occurrences
        occurrences = {}
        
        # Populate the occurrences dictionary
        for i, num in enumerate(nums):
            if num == x:
                occurrences[len(occurrences) + 1] = i
        
        # Process queries
        answer = []
        for query in queries:
            if query in occurrences:
                answer.append(occurrences[query])
            else:
                answer.append(-1)
        
        return answer