from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        positions = []
        for idx, num in enumerate(nums):
            if num == x:
                positions.append(idx)
        answer = []
        for q in queries:
            if q <= len(positions):
                answer.append(positions[q-1])
            else:
                answer.append(-1)
        return answer