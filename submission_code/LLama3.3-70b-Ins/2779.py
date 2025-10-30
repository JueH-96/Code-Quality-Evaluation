from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        
        for index, color in queries:
            nums[index] = color
            count = 0
            for i in range(n - 1):
                if nums[i] == nums[i + 1] and nums[i] != 0:
                    count += 1
            answer.append(count)
        
        return answer