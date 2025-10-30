class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Precompute all indices where x occurs
        indices = [i for i, val in enumerate(nums) if val == x]
        
        # For each query, find the required (queries[i])^th occurrence of x
        # If it doesn't exist, return -1
        answer = []
        for q in queries:
            idx = q - 1  # Convert to 0-based index
            if 0 <= idx < len(indices):
                answer.append(indices[idx])
            else:
                answer.append(-1)
        
        return answer