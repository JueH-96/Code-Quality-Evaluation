class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Calculate (a^b mod 10)
            mod10_val = pow(a, b, 10)
            # Calculate (mod10_val^c mod m)
            result = pow(mod10_val, c, m)
            if result == target:
                good_indices.append(i)
        return good_indices