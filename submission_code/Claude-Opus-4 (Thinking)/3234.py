class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i, (a, b, c, m) in enumerate(variables):
            # First compute a^b % 10
            temp = pow(a, b, 10)
            # Then compute temp^c % m
            value = pow(temp, c, m)
            if value == target:
                result.append(i)
        return result