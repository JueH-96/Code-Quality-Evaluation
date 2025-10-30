class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Compute ( (a^b) % 10 ) ^ c ) % m
            x = pow(a, b, 10)      # (a^b) % 10
            y = pow(x, c, m)       # ((a^b) % 10)^c % m
            if y == target:
                good_indices.append(i)
        return good_indices