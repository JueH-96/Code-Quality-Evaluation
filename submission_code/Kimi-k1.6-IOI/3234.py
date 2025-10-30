class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for idx, (a, b, c, m) in enumerate(variables):
            # Compute (a^b) % 10
            mod_10 = pow(a, b, 10)
            # Compute (mod_10^c) % m
            result = pow(mod_10, c, m)
            if result == target:
                good_indices.append(idx)
        return good_indices