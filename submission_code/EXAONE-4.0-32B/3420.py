class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        indices = []
        for idx, num in enumerate(nums):
            if num == x:
                indices.append(idx)
        
        result = []
        for q in queries:
            if q <= len(indices):
                result.append(indices[q-1])
            else:
                result.append(-1)
        return result