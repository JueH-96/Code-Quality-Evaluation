class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        count = nums.count(x)
        return [-1 if count < queries[i] else i for i in range(len(queries))]