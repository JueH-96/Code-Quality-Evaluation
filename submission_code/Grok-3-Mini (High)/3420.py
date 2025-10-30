class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where the value is x
        indices_of_x = [idx for idx, val in enumerate(nums) if val == x]
        
        # Prepare the answer list
        answer = []
        
        # For each query, find the index of the k-th occurrence
        for q in queries:
            if q <= len(indices_of_x):
                answer.append(indices_of_x[q - 1])
            else:
                answer.append(-1)
        
        return answer