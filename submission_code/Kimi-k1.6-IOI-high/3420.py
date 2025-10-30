class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        occurrences = [i for i, num in enumerate(nums) if num == x]
        result = []
        for q in queries:
            if q <= len(occurrences):
                result.append(occurrences[q - 1])
            else:
                result.append(-1)
        return result