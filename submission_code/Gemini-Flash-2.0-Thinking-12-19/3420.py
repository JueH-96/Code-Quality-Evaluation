class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        answer = []
        for q in queries:
            count = 0
            index_found = -1
            for i in range(len(nums)):
                if nums[i] == x:
                    count += 1
                    if count == q:
                        index_found = i
                        break
            answer.append(index_found)
        return answer