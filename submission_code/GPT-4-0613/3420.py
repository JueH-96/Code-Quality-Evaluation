class Solution:
    def occurrencesOfElement(self, nums, queries, x):
        occurrences = []
        result = []
        for i in range(len(nums)):
            if nums[i] == x:
                occurrences.append(i)
        for query in queries:
            if query <= len(occurrences):
                result.append(occurrences[query-1])
            else:
                result.append(-1)
        return result