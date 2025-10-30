class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        indices = [i for i, num in enumerate(nums) if num == x]
        result = []
        for q in queries:
            if q > len(indices):
                result.append(-1)
            else:
                result.append(indices[q-1])
        return result