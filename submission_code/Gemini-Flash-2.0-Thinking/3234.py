class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i in range(len(variables)):
            a, b, c, m = variables[i]

            # Calculate (a^b % 10)
            term1 = pow(a, b, 10)

            # Calculate ((a^b % 10)^c) % m
            result = pow(term1, c, m)

            if result == target:
                good_indices.append(i)

        return good_indices