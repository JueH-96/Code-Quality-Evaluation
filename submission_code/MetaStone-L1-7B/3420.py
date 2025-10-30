class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        occurrences = []
        for index, num in enumerate(nums):
            if num == x:
                occurrences.append(index)
        result = []
        for q in queries:
            if len(occurrences) >= q:
                result.append(occurrences[q - 1])
            else:
                result.append(-1)
        return result