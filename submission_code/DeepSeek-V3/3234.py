class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Compute a^b mod 10
            ab_mod10 = pow(a, b, 10)
            # Compute (ab_mod10)^c mod m
            result = pow(ab_mod10, c, m)
            if result == target:
                good_indices.append(i)
        return good_indices