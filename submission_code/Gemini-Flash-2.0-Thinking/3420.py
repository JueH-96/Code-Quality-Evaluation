class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        x_indices = [i for i, num in enumerate(nums) if num == x]
        result = []
        for query in queries:
            if query <= len(x_indices):
                result.append(x_indices[query - 1])
            else:
                result.append(-1)
        return result