class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        result = []
        for i, (a, b, c, m) in enumerate(variables):
            # Calculate ((a^b % 10)^c) % m
            step1 = pow(a, b, 10)  # a^b % 10
            step2 = pow(step1, c, m)  # (step1)^c % m
            if step2 == target:
                result.append(i)
        return result