class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        indices_of_x = []
        for i in range(len(nums)):
            if nums[i] == x:
                indices_of_x.append(i)
        
        result = []
        for q in queries:
            if q > len(indices_of_x):
                result.append(-1)
            else:
                result.append(indices_of_x[q-1])
        return result