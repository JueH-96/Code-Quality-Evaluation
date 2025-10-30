class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all the indices of occurrences of x in nums
        occurrences = [i for i, num in enumerate(nums) if num == x]
        
        # Prepare the answer list
        answer = []
        
        # Process each query
        for query in queries:
            if query <= len(occurrences):
                answer.append(occurrences[query - 1])
            else:
                answer.append(-1)
        
        return answer