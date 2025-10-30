from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i, (a, b, c, m) in enumerate(variables):
            part1 = pow(a, b, 10)
            part2_mod = pow(part1, c, m)
            if part2_mod == target:
                result.append(i)
        return result