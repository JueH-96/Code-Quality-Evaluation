class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # First, find all the indices where x appears in nums
        indices = [i for i, num in enumerate(nums) if num == x]
        # Initialize the answer list
        answer = []
        # Iterate over each query
        for q in queries:
            # Check if the query is within the range of indices
            if q <= len(indices):
                answer.append(indices[q-1])
            else:
                answer.append(-1)
        return answer