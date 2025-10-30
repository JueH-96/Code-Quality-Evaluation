from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i in range(len(variables)):
            a, b, c, m = variables[i]
            part1 = pow(a, b, 10)
            part2 = pow(part1, c, m)
            if part2 == target:
                result.append(i)
        return result