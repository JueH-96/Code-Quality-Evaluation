class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        occurrences = {}
        count = 0
        for i, num in enumerate(nums):
            if num == x:
                count += 1
                occurrences[count] = i
        result = []
        for query in queries:
            if query in occurrences:
                result.append(occurrences[query])
            else:
                result.append(-1)
        return result