class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        occurrences = []
        for i, num in enumerate(nums):
            if num == x:
                occurrences.append(i)
        
        answer = []
        for q in queries:
            if q <= len(occurrences):
                answer.append(occurrences[q - 1])
            else:
                answer.append(-1)
        
        return answer